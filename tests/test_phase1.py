"""Phase 1 invariants (6)."""
from pathlib import Path
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def chunks():
    p = CHECKPOINTS / "chunks.parquet"
    if not p.exists():
        pytest.skip("chunks.parquet missing — run `python src/phase1_chunk.py` first")
    return pd.read_parquet(p)


def test_chunk_count(chunks):
    # 6 books × 10 chapters × 3 passages = 180
    assert len(chunks) == 180


def test_required_columns(chunks):
    assert {"chunk_id", "book", "chapter", "position", "text", "word_count"}.issubset(chunks.columns)


def test_chunk_ids_unique_and_contiguous(chunks):
    assert chunks["chunk_id"].is_unique
    assert chunks["chunk_id"].min() == 0
    assert chunks["chunk_id"].max() == len(chunks) - 1


def test_six_books(chunks):
    assert chunks["book"].nunique() == 6


def test_ten_chapters_per_book(chunks):
    per_book = chunks.groupby("book")["chapter"].nunique()
    assert (per_book == 10).all()


def test_word_counts_reasonable(chunks):
    # ~40-80 words per chunk by construction
    assert chunks["word_count"].min() >= 10
    assert chunks["word_count"].max() <= 120
