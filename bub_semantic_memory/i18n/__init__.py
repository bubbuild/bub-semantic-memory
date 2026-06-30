# Ported from MemPalace (MIT), see ./NOTICE
"""Minimal i18n loader for cross-language entity-pattern extraction.

Only the entity-pattern loading helpers from MemPalace's i18n module are
ported here. Higher-level CLI translation helpers (``load_lang``, ``t``,
``current_lang``) are intentionally not included because cue extraction does
not need them.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

_LANG_DIR = Path(__file__).parent


def _canonical_lang(lang: str) -> str | None:
    """Resolve a language code to its on-disk canonical filename stem.

    BCP 47 tags are case-insensitive (RFC 5646 §2.1.1), and the locale
    files mix conventions (``pt-br.json`` vs ``zh-CN.json``). Match on
    lowercase so callers can pass ``PT-BR``, ``zh-cn``, ``Pt-Br``, etc.
    Returns ``None`` if no file matches.
    """
    if not lang:
        return None
    target = lang.strip().lower()
    for path in _LANG_DIR.glob("*.json"):
        if path.stem.lower() == target:
            return path.stem
    return None


def available_languages() -> list[str]:
    """Return list of available language codes."""
    return sorted(p.stem for p in _LANG_DIR.glob("*.json"))


def _load_entity_section(lang: str) -> dict:
    """Load the raw entity section for one language. Returns {} if missing."""
    canonical = _canonical_lang(lang)
    if canonical is None:
        return {}
    lang_file = _LANG_DIR / f"{canonical}.json"
    try:
        data = json.loads(lang_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data.get("entity", {}) or {}


def _script_boundary(chars: str) -> str:
    """Build a lookaround-based word boundary expression.

    Python's built-in ``\\b`` is a transition between ``\\w`` and non-``\\w``.
    ``\\w`` covers Unicode Letter and Number categories but NOT Marks (category
    Mc/Mn), so for scripts whose words contain combining vowel signs —
    Devanagari, Arabic, Hebrew, Thai, Tamil, Burmese, Khmer — the default
    ``\\b`` drops the trailing mark.

    Locales with such scripts declare ``boundary_chars`` in their entity section
    (e.g. ``"\\\\w\\\\u0900-\\\\u097F"`` for Hindi). This function returns a
    regex fragment equivalent to ``\\b`` but where the "word" side is defined
    as any char matching ``[chars]`` rather than just ``\\w``.
    """
    return (
        rf"(?:(?<=[{chars}])(?=[^{chars}])"
        rf"|(?<=[^{chars}])(?=[{chars}])"
        rf"|^(?=[{chars}])"
        rf"|(?<=[{chars}])$)"
    )


def _expand_b(pattern: str, boundary_chars: str) -> str:
    """Replace every literal ``\\b`` in ``pattern`` with a script-aware boundary.

    ``boundary_chars`` is the inside-word character class (without brackets).
    If it's falsy, the pattern is returned unchanged so ``\\b`` keeps its
    default Python ``re`` semantics.
    """
    if not boundary_chars:
        return pattern
    return pattern.replace(r"\b", _script_boundary(boundary_chars))


def _is_cjk_boundary(boundary_chars: str) -> bool:
    """Return True if the boundary only describes the CJK Unified Ideographs block.

    Chinese and Japanese are written without word separators, so requiring a
    trailing script boundary would prevent matching names in normal running
    text (e.g. "王明" in "王明最近在做什么"). For those locales we keep a
    leading boundary but allow the match to end anywhere inside the same
    script; the pattern's own length limit ({1,2} for the given-name part)
    determines where the name ends.
    """
    return boundary_chars.strip() == r"\u4E00-\u9FFF"


def _wrap_candidate(raw_pat: str, boundary_chars: str) -> str:
    """Wrap a candidate/multi-word extraction pattern with a capture group
    and word boundaries appropriate for its locale.

    Default: ``\\b(raw)\\b``. With ``boundary_chars``: the script-aware
    equivalent, so names ending in combining marks are matched in full.

    For CJK-only boundaries the trailing boundary is relaxed to a lookahead
    for a non-ideograph or end-of-string, so surname-prefixed names can be
    extracted from unspaced text.
    """
    if not boundary_chars:
        return rf"\b({raw_pat})\b"
    if _is_cjk_boundary(boundary_chars):
        # Running CJK text has no word separators. Use a leading boundary only
        # and make the name-length quantifier non-greedy so "王明" in
        # "王明最近在做什么" is captured as a two-character name rather than
        # greedily swallowing the following ideographs.
        leading = rf"(?:^|(?<=[^{boundary_chars}]))"
        cjk_pat = raw_pat.replace("{1,2}", "{1,2}?")
        return f"{leading}({cjk_pat})"
    b = _script_boundary(boundary_chars)
    return f"{b}({raw_pat}){b}"


def _collect_entity_section(section: dict, acc: dict) -> None:
    """Merge one language's entity section into the running accumulator.

    Handles boundary expansion in-place so the caller merges already-expanded
    strings: ``candidate_patterns`` and ``multi_word_patterns`` are pre-wrapped
    with the locale's boundary (capture group included, ready to compile);
    every ``\\b`` inside person/pronoun/dialogue/project/direct patterns is
    replaced with the locale's script-aware boundary.
    """
    boundary_chars = section.get("boundary_chars")
    if section.get("candidate_pattern"):
        acc["candidate_patterns"].append(
            _wrap_candidate(section["candidate_pattern"], boundary_chars)
        )
    if section.get("multi_word_pattern"):
        acc["multi_word_patterns"].append(
            _wrap_candidate(section["multi_word_pattern"], boundary_chars)
        )
    if section.get("direct_address_pattern"):
        acc["direct_address"].append(_expand_b(section["direct_address_pattern"], boundary_chars))
    acc["person_verbs"].extend(
        _expand_b(p, boundary_chars) for p in section.get("person_verb_patterns", [])
    )
    acc["pronouns"].extend(
        _expand_b(p, boundary_chars) for p in section.get("pronoun_patterns", [])
    )
    acc["dialogue"].extend(
        _expand_b(p, boundary_chars) for p in section.get("dialogue_patterns", [])
    )
    acc["project_verbs"].extend(
        _expand_b(p, boundary_chars) for p in section.get("project_verb_patterns", [])
    )
    acc["stopwords"].update(w.lower() for w in section.get("stopwords", []))


def _dedupe(items: list) -> list:
    """Remove duplicates while preserving first-occurrence order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@functools.lru_cache(maxsize=32)
def get_entity_patterns(languages=("en",)) -> dict:
    """Return merged entity detection patterns for the requested languages.

    Entity detection patterns live under each locale's ``entity`` section.
    This function merges them into a single dict for consumption by cue
    extraction.

    Merge rules:
      - List fields (person_verb_patterns, pronoun_patterns, dialogue_patterns,
        project_verb_patterns) are concatenated in the order of ``languages``,
        with duplicates removed while preserving first occurrence.
      - ``stopwords`` is the set union across all languages, returned as a
        sorted list.
      - ``candidate_patterns`` and ``multi_word_patterns`` are returned as
        **fully-wrapped regex strings** (boundary + capture group applied);
        the consumer compiles them directly with no further wrapping.
      - ``direct_address_pattern`` is returned as a list of per-language
        alternation patterns (not concatenated — each is applied separately).

    Locales with combining-mark scripts can declare ``boundary_chars`` in
    their entity section; every ``\\b`` inside that locale's patterns — plus
    the candidate/multi-word wrapping — is expanded to a script-aware
    lookaround boundary.

    English is always included as a fallback so that mixed-script queries
    and locales whose JSON file does not yet ship entity patterns (e.g.
    ``ja.json`` in the current upstream snapshot) still get basic ASCII/Latin
    handling and the full English stopword list.

    If ``languages`` is empty or no requested language declares entity data,
    English alone is returned.
    """
    if not languages:
        languages = ("en",)
    languages = tuple(_canonical_lang(lang) or lang for lang in languages)
    if "en" not in languages:
        languages = (*languages, "en")

    acc = {
        "candidate_patterns": [],
        "multi_word_patterns": [],
        "person_verbs": [],
        "pronouns": [],
        "dialogue": [],
        "direct_address": [],
        "project_verbs": [],
        "stopwords": set(),
    }

    found_any = False
    for lang in languages:
        section = _load_entity_section(lang)
        if not section:
            continue
        found_any = True
        _collect_entity_section(section, acc)

    if not found_any:
        _collect_entity_section(_load_entity_section("en"), acc)

    return {
        "candidate_patterns": acc["candidate_patterns"],
        "multi_word_patterns": acc["multi_word_patterns"],
        "person_verb_patterns": _dedupe(acc["person_verbs"]),
        "pronoun_patterns": _dedupe(acc["pronouns"]),
        "dialogue_patterns": _dedupe(acc["dialogue"]),
        "direct_address_patterns": acc["direct_address"],
        "project_verb_patterns": _dedupe(acc["project_verbs"]),
        "stopwords": sorted(acc["stopwords"]),
    }
