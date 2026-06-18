"""Tests for the semantic_memory plugin."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from republic import TapeContext, TapeEntry

from bub_semantic_memory.extractor import (
    _build_snapshot,
    _entries_to_text,
    _parse_llm_response,
    extract_semantics,
)
from bub_semantic_memory.hook_impl import build_semantic_context
from bub_semantic_memory.models import Entity, Relation, SemanticSnapshot
from bub_semantic_memory.store import SemanticStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Return a mock LLM that always yields a fixed JSON response with one entity."""
    llm = MagicMock()
    fixed_response = json.dumps(
        {
            "entities": [{"id": "alice", "type": "person", "name": "Alice", "metadata": {}}],
            "relations": [],
        }
    )
    llm.chat_async = AsyncMock(return_value=fixed_response)
    return llm


@pytest.fixture
def temp_storage():
    """Temporary directory for SemanticStore."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "semantic"


@pytest.fixture
def sample_entries() -> list[TapeEntry]:
    """Sample TapeEntry list representing a short conversation."""
    return [
        TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hello, I am Alice."}),
        TapeEntry(id=2, kind="message", payload={"role": "assistant", "content": "Hi Alice, how can I help?"}),
    ]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestEntityCreation:
    def test_entity_creation(self):
        entity = Entity(type="person", name="Alice")
        assert entity.type == "person"
        assert entity.name == "Alice"
        assert entity.id  # auto-generated UUID
        assert entity.metadata == {}

    def test_entity_with_metadata(self):
        entity = Entity(type="task", name="Deploy", metadata={"priority": "high"})
        assert entity.metadata["priority"] == "high"

    def test_entity_serialization(self):
        entity = Entity(type="person", name="Bob")
        data = json.loads(entity.model_dump_json())
        assert data["type"] == "person"
        assert data["name"] == "Bob"
        assert "id" in data

    def test_entity_deserialization(self):
        entity = Entity(type="person", name="Carol")
        restored = Entity.model_validate(json.loads(entity.model_dump_json()))
        assert restored == entity

    def test_entity_equality_by_id(self):
        entity_a = Entity(type="person", name="Alice")
        entity_b = Entity(id=entity_a.id, type="person", name="Alice")
        assert entity_a == entity_b

    def test_entity_hash(self):
        entity = Entity(type="concept", name="Python")
        assert hash(entity) == hash(entity.id)


class TestRelationCreation:
    def test_relation_creation(self):
        rel = Relation(from_id="id-a", to_id="id-b", type="uses")
        assert rel.from_id == "id-a"
        assert rel.to_id == "id-b"
        assert rel.type == "uses"
        assert rel.metadata == {}

    def test_relation_with_metadata(self):
        rel = Relation(from_id="id-a", to_id="id-b", type="depends_on", metadata={"since": "2024"})
        assert rel.metadata["since"] == "2024"

    def test_relation_serialization(self):
        rel = Relation(from_id="x", to_id="y", type="creates")
        data = json.loads(rel.model_dump_json())
        assert data["from_id"] == "x"
        assert data["to_id"] == "y"
        assert data["type"] == "creates"

    def test_relation_deserialization(self):
        rel = Relation(from_id="x", to_id="y", type="related_to")
        restored = Relation.model_validate(json.loads(rel.model_dump_json()))
        assert restored == rel

    def test_relation_equality(self):
        r1 = Relation(from_id="a", to_id="b", type="uses")
        r2 = Relation(from_id="a", to_id="b", type="uses")
        assert r1 == r2

    def test_relation_hash(self):
        r = Relation(from_id="a", to_id="b", type="uses")
        assert hash(r) == hash(("a", "b", "uses"))


class TestSemanticSnapshotJson:
    def test_semantic_snapshot_json_roundtrip(self):
        entity = Entity(type="person", name="Alice")
        rel = Relation(from_id=entity.id, to_id=entity.id, type="self_ref")
        snapshot = SemanticSnapshot(
            entities=(entity,),
            relations=(rel,),
            tape_id="tape-1",
            anchor_id="anchor-1",
        )
        serialized = snapshot.model_dump_json()
        restored = SemanticSnapshot.model_validate_json(serialized)

        assert restored.tape_id == snapshot.tape_id
        assert restored.anchor_id == snapshot.anchor_id
        assert len(restored.entities) == 1
        assert restored.entities[0] == entity
        assert len(restored.relations) == 1
        assert restored.relations[0] == rel

    def test_semantic_snapshot_defaults(self):
        snapshot = SemanticSnapshot(tape_id="t", anchor_id="a")
        assert snapshot.entities == ()
        assert snapshot.relations == ()
        assert snapshot.tape_id == "t"
        assert snapshot.anchor_id == "a"

    def test_semantic_snapshot_created_at_is_utc(self):
        from datetime import timezone

        snapshot = SemanticSnapshot(tape_id="t", anchor_id="a")
        assert snapshot.created_at.tzinfo is not None
        assert snapshot.created_at.tzinfo == timezone.utc


class TestSemanticStoreAppendLoad:
    @pytest.mark.asyncio
    async def test_append_and_load(self, temp_storage: Path):
        store = SemanticStore(storage_root=temp_storage)
        entity = Entity(type="person", name="Alice")
        snapshot = SemanticSnapshot(entities=(entity,), relations=(), tape_id="tape-1", anchor_id="a1")

        await store.append("tape-1", snapshot)
        loaded = await store.load("tape-1")

        assert len(loaded) == 1
        assert loaded[0].tape_id == "tape-1"
        assert len(loaded[0].entities) == 1
        assert loaded[0].entities[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_load_empty_when_no_file(self, temp_storage: Path):
        store = SemanticStore(storage_root=temp_storage)
        loaded = await store.load("nonexistent-tape")
        assert loaded == []

    @pytest.mark.asyncio
    async def test_multiple_appends_accumulate(self, temp_storage: Path):
        store = SemanticStore(storage_root=temp_storage)
        for i in range(3):
            snap = SemanticSnapshot(
                entities=(Entity(type="concept", name=f"Item{i}"),),
                relations=(),
                tape_id="tape-multi",
                anchor_id=f"a{i}",
            )
            await store.append("tape-multi", snap)

        loaded = await store.load("tape-multi")
        assert len(loaded) == 3

    @pytest.mark.asyncio
    async def test_tape_file_path(self, temp_storage: Path):
        store = SemanticStore(storage_root=temp_storage)
        path = store.tape_file_path("my-tape")
        assert path == temp_storage / "my-tape.jsonl"


class TestSemanticStoreCreatesDirectory:
    def test_store_creates_directory_automatically(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "deep" / "nested" / "semantic"
            assert not nested.exists()
            SemanticStore(storage_root=nested)
            assert nested.exists()


class TestExtractorWithMockLlm:
    @pytest.mark.asyncio
    async def test_extract_semantics_with_mock_llm(self, mock_llm, sample_entries: list[TapeEntry]):
        snapshot = await extract_semantics(
            entries=sample_entries,
            llm=mock_llm,
            tape_id="tape-1",
            anchor_id="anchor-1",
        )
        assert isinstance(snapshot, SemanticSnapshot)
        assert len(snapshot.entities) == 1
        assert snapshot.entities[0].name == "Alice"
        assert snapshot.entities[0].type == "person"
        assert snapshot.tape_id == "tape-1"
        assert snapshot.anchor_id == "anchor-1"

    @pytest.mark.asyncio
    async def test_extract_semantics_calls_llm(self, mock_llm, sample_entries: list[TapeEntry]):
        await extract_semantics(entries=sample_entries, llm=mock_llm)
        mock_llm.chat_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_semantics_llm_failure_returns_empty(self, sample_entries: list[TapeEntry]):
        failing_llm = MagicMock()
        failing_llm.chat_async = AsyncMock(side_effect=RuntimeError("LLM down"))
        snapshot = await extract_semantics(entries=sample_entries, llm=failing_llm)
        assert len(snapshot.entities) == 0
        assert len(snapshot.relations) == 0

    @pytest.mark.asyncio
    async def test_extract_semantics_invalid_json_returns_empty(self, sample_entries: list[TapeEntry]):
        bad_llm = MagicMock()
        bad_llm.chat_async = AsyncMock(return_value="not valid json {{{{")
        snapshot = await extract_semantics(entries=sample_entries, llm=bad_llm)
        assert len(snapshot.entities) == 0

    @pytest.mark.asyncio
    async def test_extract_semantics_with_relations(self, sample_entries: list[TapeEntry]):
        llm = MagicMock()
        llm.chat_async = AsyncMock(
            return_value=json.dumps(
                {
                    "entities": [
                        {"id": "alice", "type": "person", "name": "Alice", "metadata": {}},
                        {"id": "bub", "type": "concept", "name": "Bub", "metadata": {}},
                    ],
                    "relations": [
                        {"from": "alice", "to": "bub", "type": "uses", "metadata": {}}
                    ],
                }
            )
        )
        snapshot = await extract_semantics(entries=sample_entries, llm=llm)
        assert len(snapshot.entities) == 2
        assert len(snapshot.relations) == 1
        assert snapshot.relations[0].type == "uses"


class TestEmptyTapeExtraction:
    @pytest.mark.asyncio
    async def test_empty_entries_returns_empty_snapshot(self, mock_llm):
        snapshot = await extract_semantics(entries=[], llm=mock_llm, tape_id="t", anchor_id="a")
        assert len(snapshot.entities) == 0
        assert len(snapshot.relations) == 0
        mock_llm.chat_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_anchor_only_entries_returns_empty_snapshot(self):
        """Entries that produce no conversation text (only whitespace when rendered) return empty."""
        llm = MagicMock()
        llm.chat_async = AsyncMock(return_value=json.dumps({"entities": [], "relations": []}))
        # An anchor-only entry results in "[anchor] name" which yields empty entities from LLM
        entries = [TapeEntry(id=1, kind="anchor", payload={"name": "start"})]
        snapshot = await extract_semantics(entries=entries, llm=llm)
        assert len(snapshot.entities) == 0


class TestEntriesHelper:
    def test_entries_to_text_message(self):
        entries = [
            TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hello"}),
        ]
        text = _entries_to_text(entries)
        assert "[user] Hello" in text

    def test_entries_to_text_tool_call(self):
        entries = [
            TapeEntry(
                id=1,
                kind="tool_call",
                payload={"calls": [{"function": {"name": "my_tool"}}]},
            )
        ]
        text = _entries_to_text(entries)
        assert "[tool_call] my_tool" in text

    def test_entries_to_text_empty(self):
        assert _entries_to_text([]) == ""


class TestParseLlmResponse:
    def test_valid_response(self):
        raw = json.dumps({"entities": [{"id": "a"}], "relations": []})
        entities, relations = _parse_llm_response(raw)
        assert len(entities) == 1
        assert relations == []

    def test_invalid_json_returns_empty(self):
        entities, relations = _parse_llm_response("{{invalid}}")
        assert entities == []
        assert relations == []

    def test_non_dict_returns_empty(self):
        entities, relations = _parse_llm_response(json.dumps([1, 2, 3]))
        assert entities == []
        assert relations == []


class TestBuildSnapshot:
    def test_builds_entities_from_raw(self):
        raw_entities = [{"id": "alice", "type": "person", "name": "Alice", "metadata": {}}]
        snapshot = _build_snapshot(raw_entities, [], tape_id="t", anchor_id="a")
        assert len(snapshot.entities) == 1
        assert snapshot.entities[0].name == "Alice"

    def test_builds_relations_from_raw(self):
        raw_entities = [
            {"id": "alice", "type": "person", "name": "Alice", "metadata": {}},
            {"id": "bub", "type": "concept", "name": "Bub", "metadata": {}},
        ]
        raw_relations = [{"from": "alice", "to": "bub", "type": "uses", "metadata": {}}]
        snapshot = _build_snapshot(raw_entities, raw_relations, tape_id="t", anchor_id="a")
        assert len(snapshot.relations) == 1

    def test_skips_relations_with_unknown_slugs(self):
        raw_entities = [{"id": "alice", "type": "person", "name": "Alice", "metadata": {}}]
        raw_relations = [{"from": "alice", "to": "ghost", "type": "uses", "metadata": {}}]
        snapshot = _build_snapshot(raw_entities, raw_relations, tape_id="t", anchor_id="a")
        assert len(snapshot.relations) == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSemanticContextCaching:
    @pytest.mark.asyncio
    async def test_second_call_uses_cache_no_llm(self, temp_storage: Path):
        """Two calls with same entries: LLM called only once, second call uses cache."""
        from republic import TapeContext

        llm = MagicMock()
        llm.chat_async = AsyncMock(
            return_value=json.dumps(
                {"entities": [{"id": "alice", "type": "person", "name": "Alice", "metadata": {}}], "relations": []}
            )
        )
        store = SemanticStore(storage_root=temp_storage)
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hello Alice"})]

        ctx = TapeContext(state={"session_id": "tape-cache"})

        # First call — should extract
        msgs1 = await build_semantic_context(entries, ctx, llm=llm, store=store)
        assert llm.chat_async.await_count == 1

        # Second call with same entries — should use cache, no extra LLM call
        msgs2 = await build_semantic_context(entries, ctx, llm=llm, store=store)
        assert llm.chat_async.await_count == 1  # still 1
        assert msgs2 == msgs1

    @pytest.mark.asyncio
    async def test_new_entries_bypass_cache(self, temp_storage: Path):
        """When entries change (new last entry id), re-extracts."""
        from republic import TapeContext

        llm = MagicMock()
        llm.chat_async = AsyncMock(
            return_value=json.dumps(
                {"entities": [{"id": "alice", "type": "person", "name": "Alice", "metadata": {}}], "relations": []}
            )
        )
        store = SemanticStore(storage_root=temp_storage)

        ctx = TapeContext(state={"session_id": "tape-cache2"})

        entries_turn1 = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hi"})]
        entries_turn2 = [
            TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Hi"}),
            TapeEntry(id=2, kind="message", payload={"role": "assistant", "content": "Hello"}),
        ]

        await build_semantic_context(entries_turn1, ctx, llm=llm, store=store)
        assert llm.chat_async.await_count == 1

        # Different last entry → re-extracts
        await build_semantic_context(entries_turn2, ctx, llm=llm, store=store)
        assert llm.chat_async.await_count == 2


class TestSemanticContextBuilding:
    @pytest.mark.asyncio
    async def test_build_semantic_context_message_entries(self, sample_entries: list[TapeEntry]):
        context = TapeContext(select=build_semantic_context)
        messages = await build_semantic_context(sample_entries, context)

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, I am Alice."
        assert messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_build_semantic_context_with_anchor_entry(self):
        entries = [
            TapeEntry(id=1, kind="anchor", payload={"name": "start", "state": {"key": "val"}}),
            TapeEntry(id=2, kind="message", payload={"role": "user", "content": "Hi"}),
        ]
        context = TapeContext(select=build_semantic_context)
        messages = await build_semantic_context(entries, context)

        assert any(m["role"] == "assistant" and "Anchor" in m["content"] for m in messages)
        assert any(m["role"] == "user" for m in messages)

    @pytest.mark.asyncio
    async def test_build_semantic_context_empty_entries(self):
        context = TapeContext(select=build_semantic_context)
        messages = await build_semantic_context([], context)
        assert messages == []

    @pytest.mark.asyncio
    async def test_build_semantic_context_tool_call_and_result(self):
        entries = [
            TapeEntry(
                id=1,
                kind="tool_call",
                payload={"calls": [{"id": "call-1", "function": {"name": "search"}}]},
            ),
            TapeEntry(
                id=2,
                kind="tool_result",
                payload={"results": ["result text"]},
            ),
        ]
        context = TapeContext(select=build_semantic_context)
        messages = await build_semantic_context(entries, context)

        tool_call_msg = next((m for m in messages if m.get("role") == "assistant" and "tool_calls" in m), None)
        tool_result_msg = next((m for m in messages if m.get("role") == "tool"), None)

        assert tool_call_msg is not None
        assert tool_result_msg is not None
        assert tool_result_msg["tool_call_id"] == "call-1"


class TestMultiTurnSemanticMemory:
    @pytest.mark.asyncio
    async def test_multi_turn_second_turn_sees_first(self, temp_storage: Path, mock_llm):
        """Simulate two turns: first turn extracts semantic info, second turn can load it."""
        store = SemanticStore(storage_root=temp_storage)
        tape_id = "session-multi"

        # First turn: user says something about Alice
        first_turn_entries = [
            TapeEntry(id=1, kind="message", payload={"role": "user", "content": "I work with Alice on the project."}),
        ]
        snapshot_turn1 = await extract_semantics(
            entries=first_turn_entries,
            llm=mock_llm,
            tape_id=tape_id,
            anchor_id="anchor-t1",
        )
        await store.append(tape_id, snapshot_turn1)

        # Second turn: load stored snapshots and verify first turn's data is visible
        loaded_snapshots = await store.load(tape_id)
        assert len(loaded_snapshots) == 1
        assert loaded_snapshots[0].tape_id == tape_id
        assert any(e.name == "Alice" for e in loaded_snapshots[0].entities)

    @pytest.mark.asyncio
    async def test_multi_turn_accumulates_snapshots(self, temp_storage: Path):
        """Two turns both persist their snapshots; loading returns both."""
        store = SemanticStore(storage_root=temp_storage)
        tape_id = "session-accum"

        for turn_idx, name in enumerate(["Alice", "Bob"]):
            llm = MagicMock()
            llm.chat_async = AsyncMock(
                return_value=json.dumps(
                    {
                        "entities": [{"id": name.lower(), "type": "person", "name": name, "metadata": {}}],
                        "relations": [],
                    }
                )
            )
            entries = [
                TapeEntry(
                    id=turn_idx + 1,
                    kind="message",
                    payload={"role": "user", "content": f"Hello {name}"},
                )
            ]
            snapshot = await extract_semantics(
                entries=entries,
                llm=llm,
                tape_id=tape_id,
                anchor_id=f"anchor-{turn_idx}",
            )
            await store.append(tape_id, snapshot)

        loaded = await store.load(tape_id)
        assert len(loaded) == 2
        all_names = {e.name for snap in loaded for e in snap.entities}
        assert "Alice" in all_names
        assert "Bob" in all_names

    @pytest.mark.asyncio
    async def test_multi_turn_context_messages_chain(self):
        """Second turn's context should include messages from both turns."""
        turn1_entries = [
            TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Turn 1 message"}),
            TapeEntry(id=2, kind="message", payload={"role": "assistant", "content": "Turn 1 reply"}),
        ]
        turn2_entries = [
            TapeEntry(id=3, kind="message", payload={"role": "user", "content": "Turn 2 message"}),
        ]

        context = TapeContext(select=build_semantic_context)
        combined = turn1_entries + turn2_entries
        messages = await build_semantic_context(combined, context)

        assert len(messages) == 3
        contents = [m["content"] for m in messages]
        assert "Turn 1 message" in contents
        assert "Turn 1 reply" in contents
        assert "Turn 2 message" in contents
