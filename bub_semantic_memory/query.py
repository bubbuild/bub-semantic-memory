"""Query-driven semantic memory helpers.

Given a list of tape entries, derive the current user query and a set of
lowercase cue tokens used to filter which entities/relations from historical
snapshots are worth injecting into the prompt.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from republic.tape.entries import TapeEntry

from bub_semantic_memory.i18n import get_entity_patterns


def extract_query(entries: Iterable[TapeEntry]) -> str:
    """Return the text of the current user query.

    The last entry is treated as the query when it is a user message.  If it is
    not, the most recent user message is used as a fallback.  Multimodal
    content is reduced to its text parts.
    """
    entries_list = list(entries)
    if not entries_list:
        return ""

    def _message_text(entry: TapeEntry) -> str | None:
        if entry.kind != "message" or not isinstance(entry.payload, dict):
            return None
        content = entry.payload.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return None

    # Prefer the very last entry (standard context-building convention).
    if (text := _message_text(entries_list[-1])) is not None:
        return text

    # Fallback to the most recent user message.
    for entry in reversed(entries_list):
        if (
            entry.kind == "message"
            and isinstance(entry.payload, dict)
            and entry.payload.get("role") == "user"
            and (text := _message_text(entry)) is not None
        ):
            return text

    return ""


def _normalize_langs(languages) -> tuple[str, ...]:
    """Coerce a language input into a non-empty hashable tuple."""
    if not languages:
        return ("en",)
    if isinstance(languages, str):
        return (languages,)
    return tuple(languages)


def _collect_single_word_candidates(
    text: str,
    patterns: list[str],
    stopwords: set[str],
    counts: defaultdict[str, int],
) -> None:
    """Compile and run single-word candidate patterns against *text*."""
    # Case-insensitive compilation lets all-caps names (e.g. Cyrillic "АЛИСА")
    # and lower-case code terms (e.g. "deploy") be extracted and then filtered
    # by stopwords, which are already casefolded.
    for wrapped_pat in patterns:
        try:
            rx = re.compile(wrapped_pat, re.IGNORECASE)
        except re.error:
            continue
        for word in rx.findall(text):
            cf = word.casefold()
            if cf in stopwords or len(cf) < 2:
                continue
            counts[cf] += 1


def _collect_multi_word_candidates(
    text: str,
    patterns: list[str],
    stopwords: set[str],
    counts: defaultdict[str, int],
) -> None:
    """Compile and run multi-word candidate patterns against *text*."""
    for wrapped_pat in patterns:
        try:
            rx = re.compile(wrapped_pat, re.IGNORECASE)
        except re.error:
            continue
        for phrase in rx.findall(text):
            cf = phrase.casefold()
            if any(w.casefold() in stopwords for w in phrase.split()):
                continue
            counts[cf] += 1


def extract_candidates_multilang(
    text: str,
    languages: tuple[str, ...] | list[str] | str = ("en",),
    min_frequency: int = 1,
) -> set[str]:
    """Extract language-aware entity candidates from *text*.

    This is a minimal port of MemPalace's candidate extraction, scoped to cue
    generation: it returns casefolded candidate strings instead of raw counts,
    defaults to ``min_frequency=1`` (queries are short), and intentionally skips
    the COCA content-word filter and known-systems pre-pass.

    Each language contributes its own character-class pattern (ASCII for
    English, Latin+diacritics for pt-br, Cyrillic for Russian, CJK for Chinese
    and Japanese, etc.). Matches from all requested languages are unioned and
    stop-filtered.
    """
    if not text:
        return set()

    langs = _normalize_langs(languages)
    patterns = get_entity_patterns(langs)
    stopwords = set(patterns["stopwords"])

    counts: defaultdict[str, int] = defaultdict(int)
    _collect_single_word_candidates(text, patterns["candidate_patterns"], stopwords, counts)
    _collect_multi_word_candidates(text, patterns["multi_word_patterns"], stopwords, counts)

    return {name for name, count in counts.items() if count >= min_frequency}


def extract_cues(
    query: str,
    min_length: int = 3,
    languages: tuple[str, ...] | list[str] | str | None = None,
) -> set[str]:
    """Extract deterministic cue tokens from a query string.

    When *languages* is provided, language-aware candidate extraction is used
    and the result is casefolded. This filters language-specific stop words
    and keeps script-appropriate tokens (Cyrillic names, CJK surnames, etc.).

    When *languages* is ``None`` or empty, the original ASCII heuristic is used
    for backward compatibility: alphanumeric tokens at least *min_length*
    characters long, lowercased.

    Cues are used for cheap substring matching against entity names/types and
    relation types.
    """
    if not query:
        return set()

    if languages is None or not languages:
        tokens = re.findall(r"[a-zA-Z0-9]+", query)
        return {token.lower() for token in tokens if len(token) >= min_length}

    return extract_candidates_multilang(query, languages=languages)
