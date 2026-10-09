# DocRanger

TF-IDF RAG over a synthesised 6-book corpus. GenAI capstone of
[Bits to Builds](https://bitstobuilds.com). Deterministic retrieval,
cite-your-sources responses, zero LLM, zero torch — fits in **~5 MB** on the
Streamlit Cloud free tier.

**Live demo:** <https://docranger-4uua3kznwphucf3a7urrjg.streamlit.app/>
**Source:** <https://github.com/PJsAcademy/docranger>

---

## What it does

| Phase | Deliverable |
|-------|-------------|
| 1. Chunk    | Synthesise 6 themed books × 10 chapters × 3 passages = 180 chunks |
| 2. Index    | sklearn `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` |
| 3. Retrieve | Cosine top-k with 1e-6 relevance threshold + cite-your-sources template |

## Run locally

```bash
pip install -r requirements.txt
python src/phase1_chunk.py    # ~1s; writes chunks.parquet
python src/phase2_embed.py    # ~1s; writes vectorizer.pkl + tfidf_matrix.npz
python src/phase3_retrieve.py # ~1s; prints example queries
pytest tests/ -q              # 18 invariants should pass
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud (free)

```bash
REPO=docranger ../portfolio/publish.sh
```

Then at <https://share.streamlit.io>: Create app → pick this repo → main file
`streamlit_app.py` → Deploy.

## Honest limits

- **TF-IDF, not embeddings.** Synonyms miss. Methodology tab has the ONNX
  MiniLM upgrade path (~15 MB, no torch).
- **No LLM response synthesis.** Returns top passages verbatim with citations.
  Safer (no hallucination) but can't paraphrase.
- **Synth corpus.** Pipeline teaches the RAG mechanism; the specific books are
  not real literature.

## Credits

Pure-sklearn RAG shape adapted from the standard pattern. Built from the
[Bits to Builds](https://bitstobuilds.com) curriculum.
