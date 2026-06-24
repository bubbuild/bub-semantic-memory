#!/usr/bin/env python3
"""Evaluate the semantic memory plugin against the LoCoMo benchmark (ACL 2024).

Downloads locomo10.json, feeds each conversation session through the plugin
pipeline (deterministic entity extraction, no LLM), and reports per-category
recall / precision / char savings across ~1,986 QA pairs.

Usage:
    uv run python scripts/eval_locomo.py
    uv run python scripts/eval_locomo.py --download-only
    uv run python scripts/eval_locomo.py --languages en,zh-CN
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Ensure plugin is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bub_semantic_memory.context import _format_snapshots, _format_snapshots_filtered
from bub_semantic_memory.models import Entity, SemanticSnapshot
from bub_semantic_memory.query import extract_cues
from bub_semantic_memory.store import SemanticStore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "locomo10.json"
MIN_ENTITY_LEN = 3

# Category labels from LoCoMo (verified from data: 1-5)
CATEGORY_LABELS = {
    1: "adversarial",
    2: "commonsense_world",
    3: "multi_hop",
    4: "single_hop",
    5: "temporal",
}

# English stopwords (subset of MemPalace en.json entity.stopwords)
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "i", "my", "me", "he", "she", "his", "her",
    "who", "what", "when", "where", "why", "how", "which",
    "if", "then", "so", "not", "no", "yes", "ok", "okay",
    "just", "very", "really", "also", "already", "still", "even", "only",
    "here", "there", "now", "too", "up", "out", "about", "like",
    "use", "get", "got", "make", "made", "take", "put", "come", "go", "see",
    "know", "think", "true", "false", "new", "old", "all", "any", "some",
    "every", "each", "more", "less", "next", "last", "first", "second",
    "way", "time", "day", "life", "place", "thing",
    "part", "kind", "sort", "case", "point", "idea", "fact",
    "question", "answer", "reason", "number", "version",
    "hey", "hi", "hello", "thanks", "thank", "right", "let",
    "say", "said", "tell", "told", "ask", "asked", "reply", "replied",
    "well", "want", "need", "call", "called", "look", "looked",
    "thing", "things", "way", "ways", "people", "person",
    "everything", "nothing", "something", "anything",
    "everyone", "someone", "anyone", "everybody", "somebody", "nobody",
    "really", "actually", "basically", "probably", "maybe", "perhaps",
    "much", "many", "lot", "lots", "little", "few",
    "sure", "okay", "alright", "fine", "great", "good", "nice",
    "right", "wrong", "true", "false", "yes", "no",
    "oh", "ah", "um", "hmm", "uh",
    "going", "go", "went", "gone",
    "come", "came", "coming",
    "take", "took", "taking", "taken",
    "make", "made", "making",
    "give", "gave", "given", "giving",
    "let", "lets", "letting",
    "back", "still", "even", "already",
    "also", "though", "although",
    "then", "than", "else", "otherwise",
})


# ---------------------------------------------------------------------------
# Deterministic entity extraction
# ---------------------------------------------------------------------------


def _extract_entities_from_text(text: str) -> list[Entity]:
    """Extract significant entities from conversation text (no LLM).

    Returns a list of Entity objects with deterministic, lowercase names.
    Only uses regex + stopword filtering — no LLM, no external NLP.
    """
    words = re.findall(r"[A-Za-z]+", text)
    seen: set[str] = set()
    entities: list[Entity] = []

    for w in words:
        lower = w.lower()
        if len(lower) < MIN_ENTITY_LEN:
            continue
        if lower in _STOPWORDS:
            continue
        if lower in seen:
            continue
        seen.add(lower)

        # Proper noun (first-letter capital, rest lowercase or mixed case)
        if w[0].isupper() and len(w) >= 2:
            entities.append(Entity(type="person" if len(w) <= 8 else "concept", name=w))
        # Significant lowercase word (length >= 4, not stopword)
        elif len(w) >= 4:
            entities.append(Entity(type="concept", name=w))

    return entities


def _target_entities_from_answer(answer: str) -> set[str]:
    """Extract target entity names from a QA answer text."""
    words = re.findall(r"[A-Za-z]+", answer)
    targets: set[str] = set()
    for w in words:
        lower = w.lower()
        if len(lower) >= MIN_ENTITY_LEN and lower not in _STOPWORDS:
            targets.add(lower)
    return targets


def _build_conversation_snapshots(
    conversation: dict,
    tape_id: str,
) -> tuple[list[SemanticSnapshot], list[str]]:
    """Build one snapshot per conversation session.
    Returns (snapshots, session_texts) for later comparison.
    """
    snapshots: list[SemanticSnapshot] = []
    session_texts: list[str] = []

    # Collect and sort session keys (exclude metadata like session_N_date_time)
    sess_keys = sorted(
        (k for k in conversation if re.match(r"session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    )

    for sidx, skey in enumerate(sess_keys):
        turns = conversation[skey]
        if not isinstance(turns, list):
            continue

        # Concatenate all dialogue turns
        text_parts: list[str] = []
        for turn in turns:
            if isinstance(turn, dict):
                text = turn.get("text", "") or turn.get("content", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
                elif isinstance(text, list):
                    for part in text:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
            elif isinstance(turn, str) and turn.strip():
                text_parts.append(turn)
        session_text = "\n".join(text_parts)
        session_texts.append(session_text)

        entities = _extract_entities_from_text(session_text)
        snap = SemanticSnapshot(
            entities=tuple(entities),
            relations=(),
            tape_id=tape_id,
            anchor_id=skey,
        )
        snapshots.append(snap)

    return snapshots, session_texts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_qa(
    query_text: str,
    answer_text: str,
    snapshots: list[SemanticSnapshot],
    languages: tuple[str, ...],
) -> dict:
    """Evaluate baseline vs query-driven for one QA pair.

    Returns dict with recall, precision, char_savings_pct, metrics.
    """
    target_entities = _target_entities_from_answer(str(answer_text))

    if not snapshots:
        return {
            "recall": 0.0, "precision": 1.0,
            "baseline_chars": 0, "query_chars": 0,
            "char_savings_pct": 0.0,
            "target_count": len(target_entities),
            "kept_count": 0,
            "relevant_kept": 0,
        }

    # Baseline: full view
    baseline_block = _format_snapshots(snapshots)
    baseline_chars = len(baseline_block)

    # Query-driven: filtered by cues
    cues = extract_cues(query_text, languages=languages)
    if cues:
        query_block = _format_snapshots_filtered(snapshots, cues)
    else:
        # No cues → fallback to full formatter
        query_block = baseline_block
    query_chars = len(query_block)

    # Compute recall and precision against target entities
    query_lower = query_block.lower()
    relevant_kept = sum(1 for t in target_entities if t in query_lower)
    kept_entity_names = set(re.findall(r"concept:|person:", query_lower))

    kept_count = query_lower.count("concept:") + query_lower.count("person:")
    recall = relevant_kept / max(len(target_entities), 1)
    # Precision: fraction of kept entities that are target entities
    kept_total = len(target_entities & {e.lower() for e in kept_entity_names}) if kept_entity_names else 0
    kept_sum = kept_count + relevant_kept  # rough proxy

    # Actually compute kept properly
    kept_names_in_block: set[str] = set()
    for line in query_block.split("\n"):
        if line.startswith("- person:") or line.startswith("- concept:"):
            name = line.split("(")[0].split(":")[-1].strip()
            kept_names_in_block.add(name.lower())

    if kept_names_in_block:
        relevant_kept_count = sum(1 for t in target_entities if t in query_block.lower())
        precision = relevant_kept_count / max(len(kept_names_in_block), 1)
    else:
        precision = 1.0 if not target_entities else 0.0

    char_savings_pct = round(
        (1 - query_chars / max(baseline_chars, 1)) * 100, 1
    )

    return {
        "recall": round(relevant_kept / max(len(target_entities), 1), 3),
        "precision": round(precision, 3),
        "baseline_chars": baseline_chars,
        "query_chars": query_chars,
        "char_savings_pct": char_savings_pct,
        "token_savings": round((baseline_chars - query_chars) / 4, 0),
        "target_count": len(target_entities),
        "relevant_kept": relevant_kept,
        "kept_count": len(kept_names_in_block),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def download_data(force: bool = False) -> Path:
    """Download LoCoMo dataset if not cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 1_000_000 and not force:
        print(f"  cached: {DATA_PATH} ({DATA_PATH.stat().st_size // 1024} KB)")
        return DATA_PATH

    print(f"  downloading {LOCOMO_URL}...")
    urllib.request.urlretrieve(LOCOMO_URL, DATA_PATH)
    size_kb = DATA_PATH.stat().st_size // 1024
    print(f"  saved: {DATA_PATH} ({size_kb} KB)")
    return DATA_PATH


def run_eval(languages: tuple[str, ...]) -> float:
    """Run evaluation over all LoCoMo conversations."""
    print(f"Loading {DATA_PATH}...")
    with open(DATA_PATH, encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"  {len(conversations)} conversations\n")

    all_results: list[dict] = []
    cat_stats: dict[int, list[dict]] = defaultdict(list)

    for cidx, conv in enumerate(conversations):
        tape_id = f"locomo_{cidx}"
        conversation = conv.get("conversation", {})
        qa_list = conv.get("qa", [])

        if not conversation or not qa_list:
            print(f"  [{cidx + 1}/{len(conversations)}] skipped (no sessions or QA)")
            continue

        print(f"  [{cidx + 1}/{len(conversations)}] conv {cidx} — "
              f"{sum(1 for k in conversation if re.match(r'session_\d+$', k))} sessions, {len(qa_list)} QA pairs")

        # Build all snapshots for this conversation
        snapshots, _stexts = _build_conversation_snapshots(conversation, tape_id)

        if not snapshots:
            print(f"    -> no entities extracted (empty snapshots)")
            continue

        # Evaluate each QA pair against accumulated store
        # (All QAs use full conversation context)
        store = SemanticStore(storage_root=Path("/tmp/locomo_eval"))
        for snap in snapshots:
            import asyncio
            asyncio.run(store.append(tape_id, snap))

        loaded = asyncio.run(store.load(tape_id))

        for qa_idx, qa in enumerate(qa_list):
            question = qa.get("question", "")
            answer = str(qa.get("answer", "") or "")
            category = qa.get("category", 0)

            if not question or not answer:
                continue

            result = _eval_qa(question, answer, loaded, languages)
            result["conv_idx"] = cidx
            result["qa_idx"] = qa_idx
            result["category"] = category
            all_results.append(result)
            cat_stats[category].append(result)

        # Quick progress indicator
        conv_recall = sum(r["recall"] for r in all_results if r["conv_idx"] == cidx)
        conv_count = sum(1 for r in all_results if r["conv_idx"] == cidx)
        if conv_count:
            print(f"    -> running recall avg: {conv_recall / conv_count:.3f} "
                  f"({conv_count} QAs)")

    # ---- Print results ----
    print("\n" + "=" * 90)
    print("  LoCoMo Evaluation — Per-Category Results")
    print("=" * 90)
    print(f"  {'Category':<22} {'QAs':>5} {'Recall':>8} {'Precision':>11} "
          f"{'Baseline':>10} {'Query':>8} {'Saved%':>7} {'ToksSaved':>10}")
    print("  " + "-" * 90)

    overall_qas = 0
    overall_recall = 0.0
    overall_precision = 0.0
    overall_baseline = 0
    overall_query = 0

    for cat_id in sorted(cat_stats.keys()):
        results = cat_stats[cat_id]
        label = CATEGORY_LABELS.get(cat_id, f"cat_{cat_id}")
        qas = len(results)
        recall = sum(r["recall"] for r in results) / qas
        precision = sum(r["precision"] for r in results) / qas
        baseline_chars = sum(r["baseline_chars"] for r in results)
        query_chars = sum(r["query_chars"] for r in results)
        saved_pct = round((1 - query_chars / max(baseline_chars, 1)) * 100, 1)
        toks_saved = sum(r["token_savings"] for r in results)

        overall_qas += qas
        overall_recall += sum(r["recall"] for r in results)
        overall_precision += sum(r["precision"] for r in results)
        overall_baseline += baseline_chars
        overall_query += query_chars

        print(f"  {label:<22} {qas:>5}  {recall:>7.3f}  {precision:>8.3f}  "
              f"{baseline_chars:>8,}  {query_chars:>6,}  {saved_pct:>6.1f}%  "
              f"{toks_saved:>8.0f}")

    # Overall
    overall_len = len(all_results)
    overall_recall_avg = overall_recall / overall_len if overall_len else 0
    overall_precision_avg = overall_precision / overall_len if overall_len else 0
    overall_saved_pct = round(
        (1 - overall_query / max(overall_baseline, 1)) * 100, 1
    )
    overall_toks = sum(r["token_savings"] for r in all_results)

    print("  " + "-" * 90)
    print(f"  {'OVERALL':<22} {overall_qas:>5}  {overall_recall_avg:>7.3f}  "
          f"{overall_precision_avg:>8.3f}  "
          f"{overall_baseline:>8,}  {overall_query:>6,}  "
          f"{overall_saved_pct:>6.1f}%  {overall_toks:>8.0f}")
    print("=" * 90)

    return overall_recall_avg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic memory plugin against LoCoMo benchmark"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download the LoCoMo dataset, don't run evaluation"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--languages", default="en",
        help="Comma-separated BUB_SEMANTIC_LANGS (default: en)"
    )
    args = parser.parse_args()

    os.environ["BUB_SEMANTIC_LANGS"] = args.languages

    print("=" * 90)
    print("  LoCoMo Plugin Evaluation")
    print(f"  languages=[{args.languages}]")
    print("=" * 90)

    download_data(force=args.force_download)

    if args.download_only:
        print(f"  file size: {DATA_PATH.stat().st_size // 1024} KB")
        return

    run_eval(languages=tuple(l.strip() for l in args.languages.split(",")))


if __name__ == "__main__":
    main()
