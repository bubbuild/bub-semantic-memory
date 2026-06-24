"""Comprehensive pre-vs-post comparison: 8 dimensions + fallback.

Each test method exercises ``build_semantic_context`` (pre) and
``build_semantic_context_query_driven`` (post) against a real
``SemanticStore`` + mock LLM, printing actual character counts.

The final summary table (``test_ZZ_summary_table_print``) collates all 9
lines into one human-readable report card.
"""

# ruff: noqa: S101

from __future__ import annotations

import asyncio
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

# ---------------------------------------------------------------------------
# Fixtures: stores
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_llm() -> MagicMock:
    llm = MagicMock()
    llm.chat_async = AsyncMock(return_value=json.dumps({"entities": [], "relations": []}))
    return llm


@pytest.fixture
def store_6e3r() -> SemanticStore:
    """6 entities + 3 relations (used by D1, D3, D4)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        alice = Entity(type="person", name="Alice")
        projx = Entity(type="task", name="ProjectX")
        bob = Entity(type="person", name="Bob")
        db = Entity(type="concept", name="Database")
        weather = Entity(type="concept", name="Weather")
        deploy = Entity(type="task", name="deploy_task")
        snap = SemanticSnapshot(
            entities=(alice, projx, bob, db, weather, deploy),
            relations=(
                Relation(from_id=alice.id, to_id=projx.id, type="works_on"),
                Relation(from_id=bob.id, to_id=db.id, type="manages"),
                Relation(from_id=deploy.id, to_id=db.id, type="depends_on"),
            ),
            tape_id="comp-tape", anchor_id="a0",
        )
        asyncio.run(store.append("comp-tape", snap))
        yield store


@pytest.fixture
def store_zh_cn() -> SemanticStore:
    """Chinese-mixed store for D2."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        wm = Entity(type="person", name="王明")
        proj = Entity(type="task", name="项目X")
        bob = Entity(type="person", name="Bob")
        db = Entity(type="concept", name="Database")
        wth = Entity(type="concept", name="天气")
        snap = SemanticSnapshot(
            entities=(wm, proj, bob, db, wth),
            relations=(
                Relation(from_id=wm.id, to_id=proj.id, type="works_on"),
                Relation(from_id=bob.id, to_id=db.id, type="manages"),
            ),
            tape_id="zh-comp-tape", anchor_id="a0",
        )
        asyncio.run(store.append("zh-comp-tape", snap))
        yield store


@pytest.fixture
def store_quadrants() -> SemanticStore:
    """Sparse graph: 5 entities, 2 disconnected groups, for D5 1-hop test.

    Group 1 (reachable from "Alice" cue): Alice -> Bob (friend)
    Group 2 (NOT reachable): Carol -> Database (uses)
    Isolated: Weather (no relations at all)
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        alice = Entity(type="person", name="Alice")
        bob = Entity(type="person", name="Bob")
        carol = Entity(type="person", name="Carol")
        db = Entity(type="concept", name="Database")
        weather = Entity(type="concept", name="Weather")
        snap = SemanticSnapshot(
            entities=(alice, bob, carol, db, weather),
            relations=(
                # Reachable group: Alice -> Bob
                Relation(from_id=alice.id, to_id=bob.id, type="friend"),
                # Unreachable group: Carol -> Database
                Relation(from_id=carol.id, to_id=db.id, type="uses"),
            ),
            tape_id="quad-tape", anchor_id="a0",
        )
        asyncio.run(store.append("quad-tape", snap))
        yield store


@pytest.fixture
def store_large_tape() -> SemanticStore:
    """31 entities + 30 relations for D7 large-scale test."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        entities = [Entity(type="person", name=f"Person{i}") for i in range(30)]
        alice = Entity(type="person", name="Alice")
        entities.append(alice)
        relations = []
        for i in range(30):
            relations.append(
                Relation(from_id=entities[i].id, to_id=entities[(i + 1) % 30].id, type="link")
            )
        # Alice self-knows
        relations.append(
            Relation(from_id=alice.id, to_id=alice.id, type="knows")
        )
        snap = SemanticSnapshot(
            entities=tuple(entities), relations=tuple(relations),
            tape_id="big-tape", anchor_id="a0",
        )
        asyncio.run(store.append("big-tape", snap))
        yield store


@pytest.fixture
def store_d8_nomatch() -> SemanticStore:
    """Store with NO "Weather" entity — D8 cue "weather" matches nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        alice = Entity(type="person", name="Alice")
        bob = Entity(type="person", name="Bob")
        db = Entity(type="concept", name="Database")
        snap = SemanticSnapshot(
            entities=(alice, bob, db),
            relations=(),
            tape_id="no-tape", anchor_id="a0",
        )
        asyncio.run(store.append("no-tape", snap))
        yield store


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _semantic_block(messages: list[dict[str, Any]]) -> str | None:
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


# ---------------------------------------------------------------------------
# Summary collector (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _summary() -> list[str]:
    return []


# ---------------------------------------------------------------------------
# Dimension tests
# ---------------------------------------------------------------------------


class TestComprehensiveComparison:
    """8 dimensions + F0 fallback + ZZ summary.

    Note: D1/D3/D4 cues for query "Tell me about Alice and ProjectX." with
    ``languages=("en",)`` include ``{'tell', 'alice', 'projectx'}`` — ``tell``
    is not a stopword in the ported English data, but it does not match any
    entity name via substring; it is harmless.
    """

    # ---- D1: token-volume (English) ----
    @pytest.mark.asyncio
    async def test_D1_token_volume_english(
        self,
        store_6e3r: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "comp-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Tell me about Alice and ProjectX."})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_6e3r)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_6e3r)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"D1 | en 6e3r | baseline={len(b)}c | query={len(q)}c")

        print(f"\n--- D1: en 6e3r ---\nbaseline ({len(b)}c):\n{b}\nquery ({len(q)}c):\n{q}")
        assert len(q) <= len(b)

    # ---- D2: token-volume (Chinese) ----
    @pytest.mark.asyncio
    async def test_D2_token_volume_chinese(
        self,
        store_zh_cn: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "zh-CN")
        ctx = TapeContext(state={"session_id": "zh-comp-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "王明最近在做什么"})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_zh_cn)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_zh_cn)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"D2 | zh 5e2r | baseline={len(b)}c | query={len(q)}c")

        print(f"\n--- D2: zh-CN 5e2r ---\nbaseline ({len(b)}c):\n{b}\nquery ({len(q)}c):\n{q}")
        assert len(q) <= len(b)
        assert "王明" in q

    # ---- D3: recall ----
    @pytest.mark.asyncio
    async def test_D3_recall(
        self,
        store_6e3r: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "comp-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Tell me about Alice and ProjectX."})]

        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_6e3r)
        q = _semantic_block(query)
        assert q is not None

        # Cues: {'tell', 'alice', 'projectx'}. Both Alice and ProjectX match.
        assert "Alice" in q
        assert "ProjectX" in q
        # Relation works_on survives (both endpoints kept).
        assert "works_on" in q

    # ---- D4: precision ----
    @pytest.mark.asyncio
    async def test_D4_precision(
        self,
        store_6e3r: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "comp-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Tell me about Alice and ProjectX."})]

        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_6e3r)
        q = _semantic_block(query)
        assert q is not None

        # Cues: {'tell', 'alice', 'projectx'} — "tell" substring does not match any entity name.
        assert "Bob" not in q
        assert "Database" not in q
        assert "Weather" not in q
        # deploy_task: no cue substring "deploy" is present at cue level for this query sequence — verified via probe.
        assert "deploy_task" not in q
        # Relations whose endpoints are both NOT kept are dropped.
        assert "manages" not in q  # Bob→Database: neither endpoint kept
        assert "depends_on" not in q  # deploy_task→Database: neither kept

    # ---- D5: relation retention quadrants ----
    @pytest.mark.asyncio
    async def test_D5_relation_retention_1hop(
        self,
        store_quadrants: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        """D5: 1-hop relation traversal. Query "Alice";
        Alice matches cue -> kept. Bob kept via Alice->Bob (friend) 1-hop.
        Carol, Database, Weather are NOT reachable -> dropped.
        """
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "quad-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "Alice"})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_quadrants)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_quadrants)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"D5 | 5e2r sparse 1-hop | baseline={len(b)}c | query={len(q)}c")

        print(f"\n--- D5: 1-hop relation traversal ---\nbaseline ({len(b)}c):\n{b}\nquery ({len(q)}c):\n{q}")

        # Direct cue match: Alice kept.
        assert "Alice" in q
        # 1-hop: Bob kept via Alice->Bob (friend) relation.
        assert "Bob" in q
        assert "friend" in q

        # Disconnected group: Carol, Database NOT reachable from Alice.
        assert "Carol" not in q
        assert "Database" not in q
        assert "uses" not in q  # Carol->Database relation dropped

        # Isolated entity: Weather not reachable.
        assert "Weather" not in q

        # Baseline: full view has everything.
        assert "Carol" in b
        assert "Database" in b
        assert "Weather" in b

    # ---- D6: multi-turn accumulation growth ----
    @pytest.mark.asyncio
    async def test_D6_multi_turn_growth(
        self,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "What about Alice?"})]

        # Build and test at each snapshot count — use async await, not asyncio.run().
        baseline_chars: dict[int, int] = {}
        query_chars: dict[int, int] = {}
        for n in [1, 2, 4, 8]:
            with tempfile.TemporaryDirectory() as tmp:
                store = SemanticStore(storage_root=Path(tmp))
                for i in range(n):
                    e = Entity(type="person", name=names[i])
                    r = Relation(from_id=e.id, to_id=e.id, type="knows")
                    snap = SemanticSnapshot(
                        entities=(e,), relations=(r,),
                        tape_id=f"d6-tape-{n}", anchor_id=f"a{i}",
                    )
                    await store.append(f"d6-tape-{n}", snap)

                ctx = TapeContext(state={"session_id": f"d6-tape-{n}"})
                baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store)
                query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store)
                b, q = _semantic_block(baseline), _semantic_block(query)
                assert b is not None and q is not None
                baseline_chars[n] = len(b)
                query_chars[n] = len(q)

        # Print growth curve and add each to summary.
        for n in [1, 2, 4, 8]:
            line = f"D6 | n={n} snapshots | baseline={baseline_chars[n]}c | query={query_chars[n]}c"
            _summary.append(line)
            print(line)

        # Assert baseline grows with snapshot count.
        assert baseline_chars[8] > baseline_chars[4] > baseline_chars[2] > baseline_chars[1],             "baseline should grow monotonically with snapshot count"
        # Assert query-driven stays bounded (only Alice matches cue "alice").
        assert query_chars[8] <= query_chars[1] + 5,             "query-driven should stay near-constant (only Alice matches)"

    # ---- D7: large-tape scale ----
    @pytest.mark.asyncio
    async def test_D7_large_tape_scale(
        self,
        store_large_tape: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "big-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "What about Alice?"})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_large_tape)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_large_tape)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"D7 | 31e30r large | baseline={len(b)}c | query={len(q)}c")

        print(f"\n--- D7: 31e30r large tape ---\nbaseline ({len(b)}c):\n{b[:300]}...\nquery ({len(q)}c):\n{q}")

        assert len(b) > 2000  # baseline above 2k characters for 31 entities
        assert len(q) < 200   # query-driven well under 200 chars (only Alice+knows)
        assert "Alice" in q
        assert "knows" in q  # relation survives (both endpoints = Alice kept)
        assert "Person15" not in q  # no Person entity survives (no "alice" in name)

    # ---- D8: no-match query degeneration ----
    @pytest.mark.asyncio
    async def test_D8_no_match_query(
        self,
        store_d8_nomatch: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "en")
        ctx = TapeContext(state={"session_id": "no-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "What is the weather?"})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_d8_nomatch)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_d8_nomatch)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"D8 | no-match | baseline={len(b)}c | query={len(q)}c")

        print(f"\n--- D8: no-match query ---\nbaseline ({len(b)}c):\n{b}\nquery ({len(q)}c):\n{q}")

        # Cue "weather" is present, but no entity contains "weather" → empty block.
        assert "Entities (0)" in q
        assert "Relations (0)" in q
        assert len(q) < len(b)

    # ---- F0: no-cues fallback ----
    @pytest.mark.asyncio
    async def test_F0_no_cues_falls_back_to_full(
        self,
        store_6e3r: SemanticStore,
        empty_llm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        _summary: list[str],
    ) -> None:
        """Pure-Chinese query without BUB_SEMANTIC_LANGS → ASCII fallback yields empty cues → full formatter."""
        monkeypatch.delenv("BUB_SEMANTIC_LANGS", raising=False)
        ctx = TapeContext(state={"session_id": "comp-tape"})
        entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": "王明最近在做什么"})]

        baseline = await build_semantic_context(entries, ctx, llm=empty_llm, store=store_6e3r)
        query = await build_semantic_context_query_driven(entries, ctx, llm=empty_llm, store=store_6e3r)

        b, q = _semantic_block(baseline), _semantic_block(query)
        assert b is not None and q is not None

        _summary.append(f"F0 | fallback no-cues | baseline={len(b)}c | query={len(q)}c | FALLBACK_EQUAL")

        print(f"\n--- F0: no-cues fallback ---\nbaseline ({len(b)}c):\n{b}\nquery ({len(q)}c):\n{q}")

        # Query "王明最近在做什么" with ASCII fallback (no BUB_SEMANTIC_LANGS=zh-CN)
        # → extract_cues returns set() → _format_snapshots_filtered delegates to _format_snapshots.
        assert len(q) == len(b)

    # ---- ZZ: summary table (runs last) ----
    def test_ZZ_summary_table_print(
        self,
        _summary: list[str],
    ) -> None:
        """Print the comparison report card."""
        print("\n\n" + "=" * 64)
        print("  COMPREHENSIVE PRE-vs-POST COMPARISON REPORT")
        print("=" * 64)
        print(f"  {'Dimension':<8} {'Scenario':<22} {'Pre (chars)':<15} {'Post (chars)':<15}")
        print("  " + "-" * 60)
        for line in _summary:
            print(f"  {line}")
        print("=" * 64)
