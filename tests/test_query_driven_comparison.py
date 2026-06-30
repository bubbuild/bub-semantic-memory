"""Compare baseline semantic memory with the query-driven variant."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from bub_semantic_memory.hook_impl import (
    build_semantic_context,
    build_semantic_context_query_driven,
)
from bub_semantic_memory.models import Entity, Relation, SemanticSnapshot
from bub_semantic_memory.store import SemanticStore
from republic import TapeContext, TapeEntry


@pytest.fixture
def comparison_store():
    """A fresh store pre-populated with mixed relevant/irrelevant snapshots."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))

        alice = Entity(type="person", name="Alice")
        bob = Entity(type="person", name="Bob")
        project_x = Entity(type="task", name="ProjectX")
        deploy_task = Entity(type="task", name="deploy_task")
        database = Entity(type="concept", name="Database")
        weather = Entity(type="concept", name="Weather")

        snapshot = SemanticSnapshot(
            entities=(alice, bob, project_x, deploy_task, database, weather),
            relations=(
                Relation(from_id=alice.id, to_id=project_x.id, type="works_on"),
                Relation(from_id=bob.id, to_id=database.id, type="manages"),
                Relation(from_id=deploy_task.id, to_id=database.id, type="depends_on"),
            ),
            tape_id="comparison-tape",
            anchor_id="anchor-0",
        )

        import asyncio

        asyncio.run(store.append("comparison-tape", snapshot))
        yield store


@pytest.fixture
def empty_llm():
    """Mock LLM that extracts no new entities/relations from the current turn."""
    llm = MagicMock()
    llm.chat_async = AsyncMock(return_value=json.dumps({"entities": [], "relations": []}))
    return llm


@pytest.fixture
def query_entries() -> list[TapeEntry]:
    """A turn whose query only mentions Alice and ProjectX."""
    return [
        TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hi, I am Bob."}),
        TapeEntry(
            id=2,
            kind="message",
            payload={"role": "assistant", "content": "Hello Bob, what do you need?"},
        ),
        TapeEntry(
            id=3,
            kind="message",
            payload={"role": "user", "content": "Tell me about Alice and ProjectX."},
        ),
    ]


class TestQueryDrivenComparison:
    @pytest.mark.asyncio
    async def test_query_driven_is_smaller_than_baseline(
        self,
        comparison_store: SemanticStore,
        empty_llm: MagicMock,
        query_entries: list[TapeEntry],
    ):
        context = TapeContext(state={"session_id": "comparison-tape"})

        baseline_msgs = await build_semantic_context(
            query_entries, context, llm=empty_llm, store=comparison_store
        )
        query_msgs = await build_semantic_context_query_driven(
            query_entries, context, llm=empty_llm, store=comparison_store
        )

        baseline_block = self._semantic_block(baseline_msgs)
        query_block = self._semantic_block(query_msgs)

        assert baseline_block is not None, "baseline should include a semantic memory block"
        assert query_block is not None, "query-driven should include a semantic memory block"

        print("\n--- baseline ---")
        print(f"chars={len(baseline_block)}\n{baseline_block}")
        print("\n--- query-driven ---")
        print(f"chars={len(query_block)}\n{query_block}")

        assert len(query_block) <= len(baseline_block)

    @pytest.mark.asyncio
    async def test_query_driven_keeps_relevant_entities(
        self,
        comparison_store: SemanticStore,
        empty_llm: MagicMock,
        query_entries: list[TapeEntry],
    ):
        context = TapeContext(state={"session_id": "comparison-tape"})

        query_msgs = await build_semantic_context_query_driven(
            query_entries, context, llm=empty_llm, store=comparison_store
        )
        query_block = self._semantic_block(query_msgs)
        assert query_block is not None

        assert "Alice" in query_block
        assert "ProjectX" in query_block
        # Relation between the two kept entities should survive filtering.
        assert "works_on" in query_block

    @pytest.mark.asyncio
    async def test_query_driven_drops_irrelevant_entities(
        self,
        comparison_store: SemanticStore,
        empty_llm: MagicMock,
        query_entries: list[TapeEntry],
    ):
        context = TapeContext(state={"session_id": "comparison-tape"})

        query_msgs = await build_semantic_context_query_driven(
            query_entries, context, llm=empty_llm, store=comparison_store
        )
        query_block = self._semantic_block(query_msgs)
        assert query_block is not None

        assert "Bob" not in query_block
        assert "Database" not in query_block
        assert "Weather" not in query_block
        assert "manages" not in query_block
        assert "depends_on" not in query_block

    @pytest.mark.asyncio
    async def test_query_driven_falls_back_when_no_cues(
        self,
        comparison_store: SemanticStore,
        empty_llm: MagicMock,
    ):
        """A query with no extractable cues should not lose information."""
        entries = [
            TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hi."}),
        ]
        context = TapeContext(state={"session_id": "comparison-tape"})

        baseline_msgs = await build_semantic_context(
            entries, context, llm=empty_llm, store=comparison_store
        )
        query_msgs = await build_semantic_context_query_driven(
            entries, context, llm=empty_llm, store=comparison_store
        )

        baseline_block = self._semantic_block(baseline_msgs)
        query_block = self._semantic_block(query_msgs)

        assert baseline_block == query_block

    def _semantic_block(self, messages: list[dict[str, Any]]) -> str | None:
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None
