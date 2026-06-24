"""End-to-end multilang comparison tests for semantic memory.

Covers the zh-CN happy-path (baseline vs query-driven character reduction with
real SemanticStore + mock LLM) and documents known-unsolvable multi-language
limitations via ``xfail strict=False`` anchors.
"""

# ruff: noqa: S101, RUF001

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
from bub_semantic_memory.query import extract_cues
from bub_semantic_memory.store import SemanticStore
from republic import TapeContext, TapeEntry


@pytest.fixture
def comparison_store_zh_cn() -> SemanticStore:
    with tempfile.TemporaryDirectory() as tmp:
        store = SemanticStore(storage_root=Path(tmp))
        wangming = Entity(type="person", name="王明")
        project = Entity(type="task", name="项目X")
        bob = Entity(type="person", name="Bob")
        database = Entity(type="concept", name="Database")
        weather = Entity(type="concept", name="天气")
        rel_wm_proj = Relation(from_id=wangming.id, to_id=project.id, type="works_on")
        rel_bob_db = Relation(from_id=bob.id, to_id=database.id, type="manages")
        snapshot = SemanticSnapshot(
            entities=(wangming, project, bob, database, weather),
            relations=(rel_wm_proj, rel_bob_db),
            tape_id="zh-tape", anchor_id="anchor-0",
        )
        asyncio.run(store.append("zh-tape", snapshot))
        yield store


@pytest.fixture
def empty_llm() -> MagicMock:
    llm = MagicMock()
    llm.chat_async = AsyncMock(return_value=json.dumps({"entities": [], "relations": []}))
    return llm


@pytest.fixture
def zh_query_entries() -> list[TapeEntry]:
    return [
        TapeEntry(id=1, kind="message", payload={"role": "user", "content": "我叫张三"}),
        TapeEntry(id=2, kind="message", payload={"role": "assistant", "content": "你好，张三"}),
        TapeEntry(id=3, kind="message", payload={"role": "user", "content": "王明最近在做什么"}),
    ]


def _semantic_block(messages: list[dict[str, Any]]) -> str | None:
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


class TestZhCnEndToEnd:
    @pytest.mark.asyncio
    async def test_zh_cn_query_driven_is_smaller_than_baseline(
        self, comparison_store_zh_cn: SemanticStore, empty_llm: MagicMock,
        zh_query_entries: list[TapeEntry], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "zh-CN")
        context = TapeContext(state={"session_id": "zh-tape"})
        baseline_msgs = await build_semantic_context(
            zh_query_entries, context, llm=empty_llm, store=comparison_store_zh_cn
        )
        query_msgs = await build_semantic_context_query_driven(
            zh_query_entries, context, llm=empty_llm, store=comparison_store_zh_cn
        )
        baseline_block = _semantic_block(baseline_msgs)
        query_block = _semantic_block(query_msgs)
        assert baseline_block is not None
        assert query_block is not None
        print("\n--- baseline (zh-CN) ---")
        print(f"chars={len(baseline_block)}\n{baseline_block}")
        print("\n--- query-driven (zh-CN) ---")
        print(f"chars={len(query_block)}\n{query_block}")
        assert len(query_block) <= len(baseline_block)

    @pytest.mark.asyncio
    async def test_zh_cn_query_driven_keeps_relevant_drops_irrelevant(
        self, comparison_store_zh_cn: SemanticStore, empty_llm: MagicMock,
        zh_query_entries: list[TapeEntry], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BUB_SEMANTIC_LANGS", "zh-CN")
        context = TapeContext(state={"session_id": "zh-tape"})
        query_msgs = await build_semantic_context_query_driven(
            zh_query_entries, context, llm=empty_llm, store=comparison_store_zh_cn
        )
        query_block = _semantic_block(query_msgs)
        assert query_block is not None
        assert "王明" in query_block
        assert "Bob" not in query_block
        assert "Database" not in query_block
        assert "天气" not in query_block
        # 1-hop: 王明→项目X (works_on) keeps 项目X via relation traversal
        assert "项目X" in query_block
        assert "works_on" in query_block


@pytest.mark.xfail(
    reason="ru.json entity.stopwords upstream-lacks general question words (что/как/почему); see MemPalace upstream commit 65fa1517a1ebd921d346812b0322dce9a39519df",
    strict=False,
)
def test_ru_what_not_in_cues_xfail() -> None:
    cues = extract_cues("Что делает АЛИСА?", languages=("ru",))
    assert "что" not in cues


@pytest.mark.xfail(
    reason="ru.json entity.stopwords upstream-lacks common verbs (делает/делать/делал); see MemPalace upstream commit 65fa1517a1ebd921d346812b0322dce9a39519df",
    strict=False,
)
def test_ru_does_not_in_cues_xfail() -> None:
    cues = extract_cues("Что делает АЛИСА?", languages=("ru",))
    assert "делает" not in cues
