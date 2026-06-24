#!/usr/bin/env python3
"""Benchmark: simulate N-turn accumulation, compare baseline vs query-driven.

Usage:
    uv run python scripts/bench_semantic_memory.py --turns 20 --languages en,zh-CN
    uv run python scripts/bench_semantic_memory.py --query "Alice ProjectX" --languages en
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from republic import TapeContext, TapeEntry

from bub_semantic_memory.hook_impl import (
    build_semantic_context,
    build_semantic_context_query_driven,
)
from bub_semantic_memory.models import Entity, Relation, SemanticSnapshot
from bub_semantic_memory.store import SemanticStore


def _semantic_block(messages: list[dict]) -> str | None:
    for msg in messages:
        if msg.get("role") == "system" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _count_entities_in_block(block: str) -> int:
    import re
    m = re.search(r"Entities \((\d+)\)", block)
    return int(m.group(1)) if m else 0


async def run_benchmark(
    turns: int,
    query_text: str,
    languages: list[str],
    query_language: str | None,
    entity_names: list[str],
) -> list[dict]:
    """Build a store with *turns* snapshots, measure at each turn."""
    mock_llm = MagicMock()
    mock_llm.chat_async = AsyncMock(return_value=json.dumps({"entities": [], "relations": []}))

    rows: list[dict] = []

    for n in range(1, turns + 1):
        # Fresh store + fresh context each turn (avoids turn-internal caching).
        with tempfile.TemporaryDirectory() as tmp:
            store = SemanticStore(storage_root=Path(tmp))
            ctx = TapeContext(state={"session_id": "bench-tape"})
            for i in range(n):
                e = Entity(type="person", name=entity_names[i % len(entity_names)])
                r = Relation(from_id=e.id, to_id=e.id, type="knows_self")
                snap = SemanticSnapshot(
                    entities=(e,), relations=(r,),
                    tape_id="bench-tape", anchor_id=f"a{i}",
                )
                await store.append("bench-tape", snap)

            entries = [TapeEntry(id=1, kind="message", payload={"role": "user", "content": query_text})]

            # Run baseline (no language setting → ASCII fallback)
            baseline = await build_semantic_context(entries, ctx, llm=mock_llm, store=store)
            b_block = _semantic_block(baseline)

            # Run query-driven (with specified languages)
            import os
            old_env = os.environ.get("BUB_SEMANTIC_LANGS", None)
            try:
                os.environ["BUB_SEMANTIC_LANGS"] = query_language or ",".join(languages)
                query = await build_semantic_context_query_driven(entries, ctx, llm=mock_llm, store=store)
            finally:
                if old_env:
                    os.environ["BUB_SEMANTIC_LANGS"] = old_env
                else:
                    del os.environ["BUB_SEMANTIC_LANGS"]
            q_block = _semantic_block(query)

            blen = len(b_block) if b_block else 0
            qlen = len(q_block) if q_block else 0
            bent = _count_entities_in_block(b_block) if b_block else 0
            qent = _count_entities_in_block(q_block) if q_block else 0

            rows.append({
                "turn": n,
                "baseline_chars": blen,
                "query_chars": qlen,
                "saved_pct": round((1 - qlen / max(blen, 1)) * 100, 1),
                "baseline_ents": bent,
                "query_ents": qent,
            })

    return rows


def print_table(rows: list[dict], query_text: str, langs: str) -> None:
    print(f"\nBenchmark: query='{query_text}'  languages=[{langs}]")
    print(f"{'turn':>5} {'baseline_chars':>15} {'query_chars':>13} {'saved%':>8} {'baseline_ents':>15} {'query_ents':>12}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['turn']:>5}  {r['baseline_chars']:>13}  {r['query_chars']:>11}  "
            f"{r['saved_pct']:>7.1f}%  {r['baseline_ents']:>13}  {r['query_ents']:>10}"
        )
    final = rows[-1]
    print(f"\nFinal turn {final['turn']}: baseline={final['baseline_chars']}c query={final['query_chars']}c "
          f"saved={final['saved_pct']}% ents={final['baseline_ents']}->{final['query_ents']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark query-driven semantic memory")
    parser.add_argument("--turns", type=int, default=20, help="Number of conversation turns")
    parser.add_argument("--languages", default="en", help="Comma-separated language codes")
    parser.add_argument("--query", default="What about Alice?", help="User query text")
    parser.add_argument("--query-lang", default=None, help="Override BUB_SEMANTIC_LANGS for query")
    parser.add_argument("--entities", default="Alice,Bob,Carol,Dave,Eve,Frank,Grace,Heidi,Ivy,Jack",
                        help="Comma-separated entity names (cycles if fewer than turns)")
    args = parser.parse_args()

    langs = [l.strip() for l in args.languages.split(",")]
    entities = [e.strip() for e in args.entities.split(",")]

    rows = asyncio.run(run_benchmark(
        turns=args.turns,
        query_text=args.query,
        languages=langs,
        query_language=args.query_lang,
        entity_names=entities,
    ))
    print_table(rows, args.query, args.languages)


if __name__ == "__main__":
    main()
