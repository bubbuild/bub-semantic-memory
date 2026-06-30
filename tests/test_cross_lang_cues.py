"""Cross-language cue extraction tests.

These tests exercise the lightweight, deterministic, LLM-free cue extraction
introduced for multi-language queries. Two tests are deliberately marked
``xfail`` to document known-unsolvable limitations of the substring-matching
approach.
"""

# ruff: noqa: S101

from __future__ import annotations

import pytest
from bub_semantic_memory.query import extract_cues


def test_en_pronouns_and_stopwords_filtered() -> None:
    cues = extract_cues("What does Alice do?", languages=("en",))
    assert "alice" in cues
    assert "what" not in cues
    assert "does" not in cues


def test_en_casefold_german_eszett() -> None:
    cues = extract_cues("Straße", languages=("de",))
    assert any(c.casefold() == "strasse" for c in cues)


def test_zh_cn_stopwords_filtered() -> None:
    """Chinese candidate extraction relies on a surname-prefix list.

    A query with no surname yields few or no cues; this test only verifies
    that common stop words are not surfaced as cues.
    """
    cues = extract_cues("她最近在忙什么", languages=("zh-CN",))
    assert "她" not in cues
    assert "在" not in cues
    assert "什么" not in cues


def test_zh_cn_name_extracted() -> None:
    cues = extract_cues("王明最近在做什么", languages=("zh-CN",))
    assert "王明" in cues


def test_ru_cyrillic_casefold() -> None:
    """Russian all-caps names are normalized via case-insensitive matching.

    The upstream ru.json stopword list does not include ``что``/``делает``,
    so this test only asserts that the entity ``АЛИСА`` survives.
    """
    cues = extract_cues("Что делает АЛИСА?", languages=("ru",))
    assert "алиса" in cues


def test_mixed_zh_en_cjk_ascii() -> None:
    cues = extract_cues(
        "Alice 最近在 deploy ProjectX",
        languages=("en", "zh-CN"),
    )
    assert {"alice", "deploy", "projectx"}.issubset(cues)


def test_fallback_no_languages() -> None:
    cues = extract_cues("Alice recent")
    assert cues == {"alice", "recent"}


def test_empty_query_returns_empty() -> None:
    assert extract_cues("", languages=("en",)) == set()


def test_unknown_lang_falls_back_to_en() -> None:
    cues = extract_cues("Alice", languages=("xx-xx",))
    assert "alice" in cues


@pytest.mark.xfail(
    reason="ja.json ships no entity section in the ported upstream snapshot",
    strict=False,
)
def test_ja_particles_filtered() -> None:
    """Japanese is not supported by the current upstream data file.

    The ported ``ja.json`` does not contain an ``entity`` section, so kana
    candidates cannot be extracted. This test is kept as an xfail anchor so
    a future upstream update that adds Japanese patterns flips to XPASS.
    """
    cues = extract_cues(
        "アリスさんはプロジェクトを作っています",
        languages=("ja",),
    )
    assert "は" not in cues
    assert "を" not in cues
    assert any("アリス" in c for c in cues)
    assert any("プロジェクト" in c for c in cues)


@pytest.mark.xfail(
    reason="L3: pronoun reference needs offline rewrite, unsolvable at retrieval",
    strict=False,
)
def test_pronoun_reference_unsolvable() -> None:
    """Historical entity "Alice"; query uses a pronoun only.

    The cue path cannot resolve "她" -> Alice because the literal string
    "Alice" never appears in the query.
    """
    cues = extract_cues("我朋友她最近在忙什么", languages=("zh-CN",))
    assert "alice" in cues


@pytest.mark.xfail(
    reason="L5: cross-language synonym needs embedding, unsolvable by deterministic patterns",
    strict=False,
)
def test_cross_language_synonym_unsolvable() -> None:
    """Historical entity named "script"; query uses Chinese "剧本".

    Without an embedding or bilingual lexicon, the deterministic extractor
    cannot bridge the synonym gap.
    """
    cues = extract_cues("她的剧本", languages=("zh-CN",))
    assert any(c.casefold() == "script" for c in cues)
