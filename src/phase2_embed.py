"""Phase 2 — DocRanger TF-IDF indexing.

Fits sklearn TfidfVectorizer on the chunks, writes vectorizer.pkl + sparse
tfidf_matrix.npz. TF-IDF chosen over neural embeddings for the free-tier
build: ~5 MB total, no torch/transformers (which inflate the deploy by
~500 MB and slow cold-start by 2-5 minutes).

Honest limit: TF-IDF misses synonymy and paraphrase; "detective" won't match
"investigator". The honest upgrade is a quantised MiniLM ONNX model (~15 MB)
run via onnxruntime for semantic retrieval — see Methodology tab.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


def main() -> None:
    chunks = pd.read_parquet(CHECKPOINTS / "chunks.parquet")
    print(f"[phase2] loaded {len(chunks):,} chunks")

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
    )
    X = vec.fit_transform(chunks["text"].values)
    print(f"[phase2] TF-IDF matrix: {X.shape[0]:,} chunks × {X.shape[1]:,} terms "
          f"({X.nnz:,} non-zero, density {100*X.nnz/(X.shape[0]*X.shape[1]):.2f}%)")

    joblib.dump(vec, CHECKPOINTS / "vectorizer.pkl")
    sparse.save_npz(CHECKPOINTS / "tfidf_matrix.npz", X)
    print(f"[phase2] wrote vectorizer.pkl + tfidf_matrix.npz")

    top_terms = pd.Series(vec.get_feature_names_out()).sample(10, random_state=0).tolist()
    print(f"[phase2] vocabulary sample: {top_terms}")


if __name__ == "__main__":
    main()
