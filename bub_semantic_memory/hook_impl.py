"""Semantic memory plugin hook implementation."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from typing import Any

from republic import LLM, TapeContext, TapeEntry

from bub.builtin.settings import load_settings
from bub.framework import BubFramework
from bub.hookspecs import hookimpl
from bub_semantic_memory.query import extract_cues, extract_query
from bub_semantic_memory.store import SemanticStore

logger = logging.getLogger(__name__)


def _build_default_context(
    entries: Iterable[TapeEntry],
) -> list[dict[str, Any]]:
    """Build default tape context (sync helper)."""
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    for entry in entries:
        match entry.kind:
            case "anchor":
                payload = entry.payload
                content = (
                    f"[Anchor created: {payload.get('name')}]: "
                    f"{json.dumps(payload.get('state'), ensure_ascii=False)}"
                )
                messages.append({"role": "assistant", "content": content})
            case "message":
                payload = entry.payload
                if isinstance(payload, dict):
                    messages.append(dict(payload))
            case "tool_call":
                calls = entry.payload.get("calls")
                if isinstance(calls, list):
                    normalized = [dict(c) for c in calls if isinstance(c, dict)]
                    if normalized:
                        messages.append(
                            {"role": "assistant", "content": "", "tool_calls": normalized}
                        )
                    pending_calls = normalized
            case "tool_result":
                results = entry.payload.get("results")
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        msg: dict[str, Any] = {
                            "role": "tool",
                            "content": (
                                result
                                if isinstance(result, str)
                                else json.dumps(result, ensure_ascii=False)
                            ),
                        }
                        if i < len(pending_calls):
                            call = pending_calls[i]
                            if call_id := call.get("id"):
                                msg["tool_call_id"] = call_id
                            fn = call.get("function")
                            if isinstance(fn, dict) and (name := fn.get("name")):
                                msg["name"] = name
                        messages.append(msg)
                pending_calls = []

    return messages


async def build_semantic_context(
    entries: Iterable[TapeEntry],
    context: TapeContext,
    llm: LLM | None = None,
    store: SemanticStore | None = None,
) -> list[dict[str, Any]]:
    """Build context with semantic memory enhancements.

    If llm or store are not provided, returns just the base context.
    """
    # Build base context
    messages = _build_default_context(entries)

    # Extract and append semantic context if both llm and store are available
    if llm is None or store is None:
        return messages

    try:
        from bub_semantic_memory.context import _format_snapshots
        from bub_semantic_memory.extractor import extract_semantics

        entries_list = list(entries)
        if not entries_list:
            return messages

        # Get tape_id from context
        tape_id = context.state.get("session_id", "unknown") if hasattr(context, "state") else "unknown"

        # ponytail: cache snapshot within a turn — context.select fires every time
        # the LLM rebuilds context (3-5x per turn in tool-use loops). Skip
        # re-extraction when entries haven't changed.
        last_entry_id = entries_list[-1].id
        cache_key = f"_semantic_{tape_id}"
        cached = context.state.get(cache_key)
        if cached is not None and cached.get("last_entry_id") == last_entry_id:
            if cached.get("block"):
                messages.append(cached["block"])
            return messages

        # Extract new semantics
        snapshot = await extract_semantics(entries_list, llm, tape_id=tape_id)
        if snapshot.entities or snapshot.relations:
            await store.append(tape_id, snapshot)

        # Load all historical snapshots
        snapshots = await store.load(tape_id)
        if snapshots:
            semantic_block = _format_snapshots(snapshots)
            block_msg = {"role": "system", "content": semantic_block}
            messages.append(block_msg)
            context.state[cache_key] = {"last_entry_id": last_entry_id, "block": block_msg}
        else:
            context.state[cache_key] = {"last_entry_id": last_entry_id, "block": None}
    except Exception as e:
        # Graceful degradation: if semantic extraction fails, just use base context
        import logging
        logging.warning(f"Semantic extraction failed: {e}")

    return messages


async def build_semantic_context_query_driven(
    entries: Iterable[TapeEntry],
    context: TapeContext,
    llm: LLM | None = None,
    store: SemanticStore | None = None,
) -> list[dict[str, Any]]:
    """Build context with query-driven semantic memory.

    Works like :func:`build_semantic_context` but only injects historical
    entities and relations that match cues extracted from the current query.
    This reduces token usage while preserving recall for the current turn.
    """
    messages = _build_default_context(entries)

    if llm is None or store is None:
        return messages

    try:
        from bub_semantic_memory.context import _format_snapshots_filtered
        from bub_semantic_memory.extractor import extract_semantics

        entries_list = list(entries)
        if not entries_list:
            return messages

        tape_id = context.state.get("session_id", "unknown") if hasattr(context, "state") else "unknown"

        last_entry_id = entries_list[-1].id
        cache_key = f"_semantic_query_{tape_id}"
        cached = context.state.get(cache_key)
        if cached is not None and cached.get("last_entry_id") == last_entry_id:
            if cached.get("block"):
                messages.append(cached["block"])
            return messages

        # Extract new semantics from the current turn and persist them.
        snapshot = await extract_semantics(entries_list, llm, tape_id=tape_id)
        if snapshot.entities or snapshot.relations:
            await store.append(tape_id, snapshot)

        # Derive cues from the current user query.
        query = extract_query(entries_list)
        languages = os.environ.get("BUB_SEMANTIC_LANGS", "en").split(",")
        cues = extract_cues(query, languages=languages)

        # Load all historical snapshots, but only render the relevant subset.
        snapshots = await store.load(tape_id)
        if snapshots and cues:
            semantic_block = _format_snapshots_filtered(snapshots, cues)
            block_msg = {"role": "system", "content": semantic_block}
            messages.append(block_msg)
            context.state[cache_key] = {"last_entry_id": last_entry_id, "block": block_msg}

            # Instrumentation: report query-driven filtering performance.
            _em = re.search(r"Entities \((\d+)\)", semantic_block)
            _rm = re.search(r"Relations \((\d+)\)", semantic_block)
            kept_e = int(_em.group(1)) if _em else 0
            kept_r = int(_rm.group(1)) if _rm else 0
            total_e = sum(len(s.entities) for s in snapshots)
            total_r = sum(len(s.relations) for s in snapshots)
            logger.info(
                "query-driven cues=%s langs=%s %dent->%dent %drel->%drel %dc block",
                cues, languages, total_e, kept_e, total_r, kept_r, len(semantic_block),
            )
        elif snapshots:
            # No usable cues: fall back to the full formatter.
            from bub_semantic_memory.context import _format_snapshots

            semantic_block = _format_snapshots(snapshots)
            block_msg = {"role": "system", "content": semantic_block}
            messages.append(block_msg)
            context.state[cache_key] = {"last_entry_id": last_entry_id, "block": block_msg}

            total_e = sum(len(s.entities) for s in snapshots)
            total_r = sum(len(s.relations) for s in snapshots)
            logger.info(
                "query-driven FALLBACK (no cues) langs=%s %dent %drel %dc block",
                languages, total_e, total_r, len(semantic_block),
            )
        else:
            context.state[cache_key] = {"last_entry_id": last_entry_id, "block": None}
    except Exception as e:
        import logging

        logging.warning(f"Query-driven semantic extraction failed: {e}")

    return messages


class SemanticMemoryPlugin:
    """Bub plugin that provides semantic memory via a TapeContext selector."""

    def __init__(self, framework: BubFramework) -> None:
        self.framework = framework
        settings = load_settings()
        from bub.builtin.store import EmptyTapeStore

        # ponytail: EmptyTapeStore — extractor only does one-shot chat_async(),
        # no need to accumulate tape history. Semantic data lives in SemanticStore.
        self.llm = LLM(
            settings.model,
            api_key=settings.api_key,
            api_base=settings.api_base,
            tape_store=EmptyTapeStore(),
        )
        self.store = SemanticStore()

    @hookimpl
    def build_tape_context(self) -> TapeContext:
        llm = self.llm
        store = self.store

        async def select_with_semantics(entries: Iterable[TapeEntry], context: TapeContext) -> list[dict]:
            return await build_semantic_context(entries, context, llm=llm, store=store)

        return TapeContext(select=select_with_semantics)


