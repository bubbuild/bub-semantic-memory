#!/usr/bin/env python3
"""LoCoMo LLM-judge accuracy evaluation (Mem0 protocol).

For each conversation, builds a semantic store with snapshots, then for each
QA pair asks the *answerer LLM* to answer using the memory block, and the
*judege LLM* to score correctness vs ground truth.  Both baseline (full memory
block) and query-driven (filtered) paths are measured in one run.

Usage:
    uv run python scripts/eval_locomo_judge.py
    uv run python scripts/eval_locomo_judge.py --use-real-llm --max-convs 2 --max-qas 10

Judge model (in priority):
    1. gpt-4o-mini if OPENAI_API_KEY is set (comparable to Mem0 paper)
    2. deepseek:deepseek-chat if DEEPSEEK_API_KEY is set
    3. fallback to BUB_MODEL

Returns per-category accuracy table for both baseline and query-driven.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bub_semantic_memory.context import _format_snapshots, _format_snapshots_filtered
from bub_semantic_memory.models import Entity, SemanticSnapshot
from bub_semantic_memory.query import extract_cues
from bub_semantic_memory.store import SemanticStore
from republic.tape.entries import TapeEntry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "locomo10.json"
MIN_ENTITY_LEN = 3
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / ".omo" / "evidence"

# Verified from actual data (locomo10.json):
#   1 = single_hop     (fact lookup: "What did Caroline research?")
#   2 = temporal       (time: "When did Melanie paint a sunrise?")
#   3 = commonsense    (multi-hop/world: "Would Caroline likely have Dr. Seuss books?")
#   4 = open_domain    (analysis: "What did Melanie realize after the charity race?")
#   5 = adversarial    (mentions topics not in conversation — excluded by protocol)
CATEGORY_LABELS = {
    1: "single_hop",
    2: "temporal",
    3: "commonsense",
    4: "open_domain",
    5: "adversarial",
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
    "things", "ways", "people", "person",
    "everything", "nothing", "something", "anything",
    "everyone", "someone", "anyone", "everybody", "somebody", "nobody",
    "actually", "basically", "probably", "maybe", "perhaps",
    "much", "many", "lot", "lots", "little", "few",
    "sure", "alright", "fine", "great", "good", "nice",
    "wrong", "oh", "ah", "um", "hmm", "uh",
    "going", "went", "gone",
    "came", "coming",
    "took", "taking", "taken",
    "making",
    "give", "gave", "given", "giving",
    "lets", "letting",
    "back", "though", "although",
    "than", "else", "otherwise",
})


# ---------------------------------------------------------------------------
# Deterministic entity extraction (no LLM)
# ---------------------------------------------------------------------------


def _extract_entities_from_text(text: str) -> list[Entity]:
    """Extract significant entities from conversation text (no LLM)."""
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

        if w[0].isupper() and len(w) >= 2:
            entities.append(Entity(type="person" if len(w) <= 8 else "concept", name=w))
        elif len(w) >= 4:
            entities.append(Entity(type="concept", name=w))

    return entities


def _build_conversation_snapshots(
    conversation: dict,
    tape_id: str,
) -> list[SemanticSnapshot]:
    """Build one snapshot per session using regex entity extraction."""
    snapshots: list[SemanticSnapshot] = []

    sess_keys = sorted(
        (k for k in conversation if re.match(r"session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    )

    for skey in sess_keys:
        turns = conversation[skey]
        if not isinstance(turns, list):
            continue

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

        entities = _extract_entities_from_text(session_text)
        snap = SemanticSnapshot(
            entities=tuple(entities),
            relations=(),
            tape_id=tape_id,
            anchor_id=skey,
        )
        snapshots.append(snap)

    return snapshots


async def _build_llm_snapshots(
    conversation: dict,
    tape_id: str,
    llm: Any,
) -> list[SemanticSnapshot]:
    """Build one snapshot per session using the real LLM extractor."""
    from bub_semantic_memory.extractor import extract_semantics

    sess_keys = sorted(
        (k for k in conversation if re.match(r"session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    session_dates = _get_session_dates(conversation)
    snapshots: list[SemanticSnapshot] = []

    for skey in sess_keys:
        turns = conversation.get(skey, [])
        if not isinstance(turns, list):
            continue

        # Prepend session date so extractor resolves relative time references
        date_str = session_dates.get(skey, "")
        entries: list[TapeEntry] = []
        if date_str:
            entries.append(TapeEntry(
                id=len(entries),
                kind="system",
                payload={"content": f"Session date: {date_str}"},
            ))
        for turn in turns:
            if isinstance(turn, dict):
                speaker = turn.get("speaker", "user")
                text = turn.get("text", "")
                if text:
                    role = "user" if speaker != "assistant" else "assistant"
                    entries.append(TapeEntry(
                        id=len(entries),
                        kind="message",
                        payload={"role": role, "content": text},
                    ))

        if not entries:
            continue

        try:
            snapshot = await extract_semantics(
                entries, llm, tape_id=tape_id, anchor_id=skey
            )
        except Exception as exc:
            print(f"    [WARN] LLM extraction failed for {skey}: {exc}")
            snapshot = SemanticSnapshot(
                entities=(), relations=(), tape_id=tape_id, anchor_id=skey
            )

        snapshots.append(snapshot)

    return snapshots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_session_dates(conversation: dict) -> dict[str, str]:
    """Extract session date metadata from conversation dict.

    Returns dict mapping session key (e.g. 'session_1') → date string.
    """
    dates: dict[str, str] = {}
    for key, val in conversation.items():
        if key.endswith("_date_time") and isinstance(val, str):
            sess_key = key.removesuffix("_date_time")
            dates[sess_key] = val
    return dates


def _format_snapshots_with_timeline(
    snapshots: list[SemanticSnapshot],
    session_dates: dict[str, str],
) -> str:
    """Format snapshots + session timeline for temporal reasoning.

    Appends a chronological session timeline so the answerer LLM can
    resolve relative time references ("yesterday", "last week") to
    absolute dates.
    """
    block = _format_snapshots(snapshots)
    if not session_dates:
        return block

    lines: list[str] = [block, "", "## Session Timeline", ""]
    for snap in snapshots:
        date_str = session_dates.get(snap.anchor_id, "unknown date")
        lines.append(f"- {snap.anchor_id}: {date_str}")

    return "\n".join(lines)


def _format_snapshots_filtered_with_timeline(
    snapshots: list[SemanticSnapshot],
    cues: set[str],
    session_dates: dict[str, str],
) -> str:
    """Format filtered snapshots + session timeline."""
    block = _format_snapshots_filtered(snapshots, cues)
    if not session_dates:
        return block

    lines: list[str] = [block, "", "## Session Timeline", ""]
    for snap in snapshots:
        date_str = session_dates.get(snap.anchor_id, "unknown date")
        lines.append(f"- {snap.anchor_id}: {date_str}")

    return "\n".join(lines)


def download_data(force: bool = False) -> Path:
    """Download LoCoMo dataset if not cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 1_000_000 and not force:
        print(f"  cached: {DATA_PATH} ({DATA_PATH.stat().st_size // 1024} KB)")
        return DATA_PATH

    print(f"  downloading {LOCOMO_URL}...")
    urllib.request.urlretrieve(LOCOMO_URL, DATA_PATH)  # noqa: S310
    size_kb = DATA_PATH.stat().st_size // 1024
    print(f"  saved: {DATA_PATH} ({size_kb} KB)")
    return DATA_PATH


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _build_llm(model_spec: str | None = None) -> Any:
    """Create a real LLM instance matching the plugin's pattern.

    Resolves model from:
    1. explicit *model_spec* (e.g. "openai:gpt-4o-mini")
    2. DeepSeek if DEEPSEEK_API_KEY set
    3. settings model (BUB_MODEL / openrouter:openrouter/free)
    """
    from bub.builtin.settings import load_settings
    from bub.builtin.store import EmptyTapeStore

    settings = load_settings()

    if model_spec:
        return __import__("republic").LLM(
            model_spec,
            tape_store=EmptyTapeStore(),
        )

    # Try DeepSeek first
    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if ds_key:
        return __import__("republic").LLM(
            "deepseek:deepseek-chat",
            api_key=ds_key,
            tape_store=EmptyTapeStore(),
        )

    return __import__("republic").LLM(
        settings.model,
        api_key=settings.api_key,
        api_base=settings.api_base,
        tape_store=EmptyTapeStore(),
    )


async def _ask_llm(
    llm: Any,
    question: str,
    memory_block: str,
) -> str:
    """Ask the LLM to answer a question using the memory block as context."""
    system_prompt = (
        "You are a memory assistant. Use ONLY the following semantic memory to "
        "answer the question. Infer from relationships and timeline when needed.\n\n"
        f"{memory_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely. Only say 'I don't know' if the memory contains no "
        "relevant information at all."
    )
    try:
        return await llm.chat_async(
            prompt=question,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        return f"[ERROR: {exc}]"


async def _judge_answer(
    judge_llm: Any,
    ground_truth: str,
    model_answer: str,
) -> int:
    """Judge whether model_answer is correct vs ground_truth. Returns 1 or 0."""
    prompt = (
        f"Ground truth: {ground_truth}\n"
        f"Model answer: {model_answer}\n"
        "Is the model answer essentially correct? Consider semantic equivalence "
        "— different wording with the same meaning counts as correct. "
        "Partial but substantially correct answers count as YES. "
        "Reply YES or NO."
    )
    try:
        response = await judge_llm.chat_async(prompt=prompt)
        return 1 if re.search(r"\bYES\b", response.strip().upper()) else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


async def run_eval(  # noqa: C901
    conversations: list[dict],
    max_convs: int,
    max_qas: int,
    use_real_llm: bool,
    languages: tuple[str, ...],
) -> dict:
    """Run evaluation and return results dict."""
    answerer = _build_llm()

    # Judge: prefer gpt-4o-mini, then DeepSeek, then answerer
    judge_model_spec = None
    judge_source = "same as answerer (non-comparable to Mem0)"
    if os.environ.get("OPENAI_API_KEY"):
        judge_model_spec = "openai:gpt-4o-mini"
        judge_source = "gpt-4o-mini (comparable to Mem0)"
    elif os.environ.get("DEEPSEEK_API_KEY"):
        judge_model_spec = "deepseek:deepseek-chat"
        judge_source = "deepseek:deepseek-chat (non-comparable to Mem0)"

    judge_llm = _build_llm(judge_model_spec)

    print(f"  Answerer: {answerer.model}")
    print(f"  Judge:    {judge_source}")

    # Per-category accumulators
    cat_baseline: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    cat_query: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    token_baseline = 0
    token_query = 0
    total_qas = 0

    for cidx in range(min(max_convs, len(conversations))):
        conv = conversations[cidx]
        conversation = conv.get("conversation", {})
        qa_list = conv.get("qa", [])
        tape_id = f"locomo_{cidx}"

        if not conversation or not qa_list:
            continue

        n_sessions = sum(1 for k in conversation if re.match(r"session_\d+$", k))
        print(f"\n  [{cidx + 1}/{min(max_convs, len(conversations))}] conv {cidx} "
              f"— {n_sessions} sessions, {len(qa_list)} QAs")

        # Build snapshots
        if use_real_llm:
            snapshots = await _build_llm_snapshots(conversation, tape_id, answerer)
        else:
            snapshots = _build_conversation_snapshots(conversation, tape_id)

        if not snapshots:
            print("    -> no entities extracted, skipping")
            continue

        # Extract session dates for temporal reasoning
        session_dates = _get_session_dates(conversation)

        # Load into store
        store = SemanticStore(storage_root=Path("/tmp/locomo_eval_judge"))  # noqa: S108
        for snap in snapshots:
            await store.append(tape_id, snap)
        loaded = await store.load(tape_id)

        conv_qas = 0
        for qa in qa_list:
            if max_qas > 0 and conv_qas >= max_qas:
                break

            question = qa.get("question", "")
            answer = str(qa.get("answer", "") or "")
            category = qa.get("category", 0)

            if not question or not answer:
                continue
            if category == 5:
                continue  # Exclude adversarial

            conv_qas += 1
            total_qas += 1

            # Build memory blocks (with session timeline for temporal reasoning)
            baseline_block = _format_snapshots_with_timeline(loaded, session_dates)

            cues = extract_cues(question, languages=languages)
            query_block = _format_snapshots_filtered_with_timeline(loaded, cues, session_dates) if cues else baseline_block

            token_baseline += _count_tokens(baseline_block)
            token_query += _count_tokens(query_block)

            # Answer + judge in parallel (baseline and query paths are independent)
            baseline_answer, query_answer = await asyncio.gather(
                _ask_llm(answerer, question, baseline_block),
                _ask_llm(answerer, question, query_block),
            )
            baseline_score, query_score = await asyncio.gather(
                _judge_answer(judge_llm, answer, baseline_answer),
                _judge_answer(judge_llm, answer, query_answer),
            )

            cat_baseline[category]["correct"] += baseline_score
            cat_baseline[category]["total"] += 1
            cat_query[category]["correct"] += query_score
            cat_query[category]["total"] += 1

            if total_qas <= 3:
                print(f"    QA#{total_qas} (cat{category}): "
                      f"baseline={'✓' if baseline_score else '✗'} "
                      f"query={'✓' if query_score else '✗'} "
                      f"[q='{question[:40]}...']")

        print(f"    -> {conv_qas} QAs evaluated")

    return {
        "cat_baseline": dict(cat_baseline),
        "cat_query": dict(cat_query),
        "total_qas": total_qas,
        "token_baseline": token_baseline,
        "token_query": token_query,
        "judge_source": judge_source,
        "judge_model_spec": judge_model_spec or "answerer",
    }


def print_report(results: dict) -> None:
    """Print per-category accuracy table."""
    cat_baseline = results["cat_baseline"]
    cat_query = results["cat_query"]
    token_baseline = results["token_baseline"]
    token_query = results["token_query"]

    all_cats = sorted(set(list(cat_baseline.keys()) + list(cat_query.keys())))

    print("\n" + "=" * 90)
    print("  LoCoMo LLM-Judge Accuracy")
    print(f"  Judge: {results['judge_source']}")
    print("=" * 90)
    print(f"  {'Category':<20} {'QAs':>5} {'Base Acc%':>10} {'Query Acc%':>11} "
          f"{'Delta':>7} {'Base Tok':>9} {'Qry Tok':>8}")
    print("  " + "-" * 90)

    overall_base_correct = 0
    overall_base_total = 0
    overall_query_correct = 0
    overall_query_total = 0

    for cat_id in all_cats:
        label = CATEGORY_LABELS.get(cat_id, f"cat_{cat_id}")

        bc = cat_baseline.get(cat_id, {"correct": 0, "total": 0})
        qc = cat_query.get(cat_id, {"correct": 0, "total": 0})

        b_acc = bc["correct"] / max(bc["total"], 1) * 100
        q_acc = qc["correct"] / max(qc["total"], 1) * 100
        delta = q_acc - b_acc

        overall_base_correct += bc["correct"]
        overall_base_total += bc["total"]
        overall_query_correct += qc["correct"]
        overall_query_total += qc["total"]

        print(f"  {label:<20} {bc['total']:>5}  {b_acc:>8.1f}%  {q_acc:>9.1f}%  "
              f"{delta:>+6.1f}%  {token_baseline:>7,}  {token_query:>6,}")

    overall_base_acc = overall_base_correct / max(overall_base_total, 1) * 100
    overall_query_acc = overall_query_correct / max(overall_query_total, 1) * 100
    overall_delta = overall_query_acc - overall_base_acc

    token_savings_pct = round(
        (1 - token_query / max(token_baseline, 1)) * 100, 1
    )

    print("  " + "-" * 90)
    print(f"  {'OVERALL':<20} {overall_base_total:>5}  {overall_base_acc:>8.1f}%  "
          f"{overall_query_acc:>9.1f}%  {overall_delta:>+6.1f}%  "
          f"{token_baseline:>7,}  {token_query:>6,}")
    print(f"\n  Token savings: {token_savings_pct}% "
          f"({token_baseline:,} → {token_query:,} tokens)")
    print("=" * 90)

    # Published comparison (if applicable)
    if "gpt-4o-mini" in results.get("judge_source", ""):
        print()
        print("  Comparison to published LoCoMo J-scores (gpt-4o-mini judge):")
        print(f"    Ours (baseline):      {overall_base_acc:.1f}%")
        print(f"    Ours (query-driven):  {overall_query_acc:.1f}%")
        print("    Mem0:                 67%")
        print("    Zep:                  66%")
        print("    LangMem:              58%")
        print("    A-Mem:                48%")
        print("    Full-Context ceiling: 73%")
    else:
        print()
        print("  Note: Judge is NOT gpt-4o-mini — results are NOT directly")
        print("  comparable to published Mem0/Zep/LangMem numbers.")
        print("  The baseline-vs-query-driven delta IS meaningful internally.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo LLM-judge accuracy evaluation (Mem0 protocol)"
    )
    parser.add_argument(
        "--max-convs", type=int, default=1,
        help="Max conversations to evaluate (default: 1)"
    )
    parser.add_argument(
        "--max-qas", type=int, default=0,
        help="Max QAs per conversation (default: 0 = all)"
    )
    parser.add_argument(
        "--use-real-llm", action="store_true",
        help="Use real LLM for entity extraction (default: regex)"
    )
    parser.add_argument(
        "--languages", default="en",
        help="Comma-separated BUB_SEMANTIC_LANGS (default: en)"
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Override judge model (default: auto-detect)"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download the dataset"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force re-download"
    )
    args = parser.parse_args()

    os.environ.setdefault("BUB_SEMANTIC_LANGS", args.languages)

    print("=" * 90)
    print("  LoCoMo LLM-Judge Accuracy")
    print(f"  max_convs={args.max_convs}, max_qas={args.max_qas}, "
          f"use_real_llm={args.use_real_llm}, languages=[{args.languages}]")
    if args.judge_model:
        print(f"  judge_model={args.judge_model} (override)")
    print("=" * 90)

    download_data(force=args.force_download)
    if args.download_only:
        return

    with open(DATA_PATH, encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"  {len(conversations)} conversations loaded\n")

    results = asyncio.run(run_eval(
        conversations=conversations,
        max_convs=args.max_convs,
        max_qas=args.max_qas,
        use_real_llm=args.use_real_llm,
        languages=tuple(lang.strip() for lang in args.languages.split(",")),
    ))

    print_report(results)


if __name__ == "__main__":
    main()
