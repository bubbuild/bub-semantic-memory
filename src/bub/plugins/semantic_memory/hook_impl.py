"""Semantic memory plugin hook implementation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from republic import LLM, TapeContext, TapeEntry

from bub.builtin.settings import load_settings
from bub.framework import BubFramework
from bub.hookspecs import hookimpl
from bub.plugins.semantic_memory.store import SemanticStore


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
        from bub.plugins.semantic_memory.extractor import extract_semantics
        from bub.plugins.semantic_memory.context import _format_snapshots

        entries_list = list(entries)
        if not entries_list:
            return messages

        # Get tape_id from context
        tape_id = context.state.get("session_id", "unknown") if hasattr(context, "state") else "unknown"

        # Extract new semantics
        snapshot = await extract_semantics(entries_list, llm, tape_id=tape_id)
        if snapshot.entities or snapshot.relations:
            await store.append(tape_id, snapshot)

        # Load all historical snapshots
        snapshots = await store.load(tape_id)
        if snapshots:
            semantic_block = _format_snapshots(snapshots)
            messages.append({"role": "system", "content": semantic_block})
    except Exception as e:
        # Graceful degradation: if semantic extraction fails, just use base context
        import logging
        logging.warning(f"Semantic extraction failed: {e}")

    return messages


class SemanticMemoryPlugin:
    """Bub plugin that provides semantic memory via a TapeContext selector."""

    def __init__(self, framework: BubFramework) -> None:
        self.framework = framework
        settings = load_settings()
        from republic.tape import InMemoryTapeStore

        tape_store = InMemoryTapeStore()
        self.llm = LLM(
            settings.model,
            api_key=settings.api_key,
            api_base=settings.api_base,
            tape_store=tape_store,
        )
        self.store = SemanticStore()

    @hookimpl
    def build_tape_context(self) -> TapeContext:
        llm = self.llm
        store = self.store

        async def select_with_semantics(entries: Iterable[TapeEntry], context: TapeContext) -> list[dict]:
            return await build_semantic_context(entries, context, llm=llm, store=store)

        return TapeContext(select=select_with_semantics)


