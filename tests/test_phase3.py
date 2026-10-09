"""Phase 3 invariants (7) — retrieval + response composer."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phase3_retrieve import retrieve, compose  # noqa: E402


def _skip_if_no_index():
    if not (ROOT / "checkpoints" / "vectorizer.pkl").exists():
        pytest.skip("TF-IDF index missing — run phases 1+2 first")


def test_lighthouse_query_returns_lighthouse_book():
    _skip_if_no_index()
    hits = retrieve("sea storm lighthouse", k=5)
    assert len(hits) > 0
    assert hits[0].book == "The Lighthouse Keeper"


def test_garden_query_returns_flowers_book():
    _skip_if_no_index()
    hits = retrieve("garden flowers blooming", k=3)
    assert len(hits) > 0
    assert hits[0].book == "A Study of Flowers"


def test_clockmaker_query_returns_clockmaker_book():
    _skip_if_no_index()
    hits = retrieve("brass gear pendulum workshop", k=3)
    assert len(hits) > 0
    assert hits[0].book == "The Clockmaker's Apprentice"


def test_out_of_domain_returns_empty():
    _skip_if_no_index()
    # Terms that aren't in any book's vocabulary
    hits = retrieve("pineapple elephant rocketship", k=5)
    assert hits == []


def test_empty_query_returns_empty():
    _skip_if_no_index()
    assert retrieve("", k=5) == []
    assert retrieve("   ", k=5) == []


def test_scores_sorted_descending():
    _skip_if_no_index()
    hits = retrieve("sea storm lighthouse", k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_compose_cites_each_source():
    _skip_if_no_index()
    hits = retrieve("sea storm lighthouse", k=3)
    out = compose("sea storm lighthouse", hits)
    for h in hits:
        assert h.book in out
        assert f"Chapter {h.chapter}" in out
