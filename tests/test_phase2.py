"""Phase 2 invariants (5)."""
from pathlib import Path
import joblib
import pytest
from scipy import sparse

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def vec():
    p = CHECKPOINTS / "vectorizer.pkl"
    if not p.exists():
        pytest.skip("vectorizer.pkl missing — run `python src/phase2_embed.py` first")
    # Trusted artifact: produced by src/phase2_embed.py in this repo's own build.
    return joblib.load(p)


@pytest.fixture(scope="module")
def mat():
    p = CHECKPOINTS / "tfidf_matrix.npz"
    if not p.exists():
        pytest.skip("tfidf_matrix.npz missing")
    return sparse.load_npz(p)


def test_vectorizer_fit(vec):
    assert vec.vocabulary_ is not None
    assert len(vec.vocabulary_) > 100


def test_matrix_shape_matches_chunks(mat):
    assert mat.shape[0] == 180  # chunks
    assert mat.shape[1] > 100   # vocab


def test_matrix_density_reasonable(mat):
    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    assert 0.001 < density < 0.5, f"density out of expected range: {density}"


def test_vectorizer_uses_bigrams(vec):
    terms = vec.get_feature_names_out()
    has_bigrams = any(" " in t for t in terms)
    assert has_bigrams, "no bigrams found; vectorizer should be fit with ngram_range=(1,2)"


def test_matrix_row_norms_positive(mat):
    import numpy as np
    norms = sparse.linalg.norm(mat, axis=1)
    assert (norms > 0).all(), "a chunk has all-zero TF-IDF — would break cosine retrieval"
