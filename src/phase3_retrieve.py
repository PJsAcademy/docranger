"""Phase 3 — DocRanger retrieval + template response.

Loads vectorizer + sparse TF-IDF matrix once; retrieve(query, k) returns the
top-k chunks ranked by cosine similarity. compose(query, hits) formats a
"cite your sources" response — no LLM. Deterministic, no hallucinations
possible; the price is that the response can't paraphrase, only quote.

Honest limit flagged in Methodology: the honest upgrade is LLM + the
retrieved passages as grounded context + a prompt that forbids claims not
in the context. Still deterministic retrieval, LLM only for prose synthesis.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


class Hit(NamedTuple):
    rank: int
    score: float
    chunk_id: int
    book: str
    chapter: int
    position: int
    text: str


# joblib.load deserializes pickle. Safe here because vectorizer.pkl is produced
# by this repo's own src/phase2_embed.py in the same build — never load from
# an untrusted source.
_cached = {"vec": None, "mat": None, "chunks": None}


def _ensure_loaded():
    if _cached["vec"] is None:
        _cached["vec"] = joblib.load(CHECKPOINTS / "vectorizer.pkl")
        _cached["mat"] = sparse.load_npz(CHECKPOINTS / "tfidf_matrix.npz")
        _cached["chunks"] = pd.read_parquet(CHECKPOINTS / "chunks.parquet")


def retrieve(query: str, k: int = 5) -> list[Hit]:
    """Return top-k chunks by TF-IDF cosine similarity. Empty query → empty list."""
    if not query or not query.strip():
        return []
    _ensure_loaded()
    vec = _cached["vec"]
    mat = _cached["mat"]
    chunks = _cached["chunks"]

    qvec = vec.transform([query])
    sims = linear_kernel(qvec, mat).ravel()
    # Ignore chunks with zero overlap — they're meaningless cosine ties
    nonzero = np.where(sims > 1e-6)[0]
    if len(nonzero) == 0:
        return []
    k = min(k, len(nonzero))
    top_idx = nonzero[np.argsort(sims[nonzero])[::-1][:k]]
    hits = []
    for rank, idx in enumerate(top_idx, start=1):
        row = chunks.iloc[int(idx)]
        hits.append(Hit(
            rank=rank, score=float(sims[idx]),
            chunk_id=int(row["chunk_id"]), book=row["book"],
            chapter=int(row["chapter"]), position=int(row["position"]),
            text=row["text"],
        ))
    return hits


def compose(query: str, hits: list[Hit]) -> str:
    """Format a cite-your-sources response. No prose synthesis — the LLM-free
    version returns the top passages as-is with citations. See Methodology for
    the LLM upgrade path."""
    if not hits:
        return f'No passages matched "{query}" above the relevance threshold.'
    lines = [f'Top passages for query: "{query}"', ""]
    for h in hits:
        lines.append(
            f"[{h.rank}] *{h.book}*, Chapter {h.chapter}, passage {h.position} "
            f"(score {h.score:.3f})"
        )
        lines.append(f"    {h.text}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    for q in [
        "sea storm lighthouse",
        "garden flowers growing",
        "mountain glacier climbing",
        "stars telescope observation",
        "merchant spice bargain",
        "pineapple elephant rocketship",   # out-of-domain, should return 0 or weak
    ]:
        hits = retrieve(q, k=3)
        print(f"Query: {q!r:40}  ->  {len(hits)} hits  "
              f"(top score {hits[0].score if hits else 0:.3f})")
        for h in hits:
            print(f"  [{h.rank}] {h.book}  ch{h.chapter}p{h.position}  score={h.score:.3f}")


if __name__ == "__main__":
    main()
