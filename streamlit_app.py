"""DocRanger — RAG over a synthesised 6-book corpus. GenAI capstone of Bits to Builds.

TF-IDF retrieval over 180 chunks. No neural embeddings (keeps the build under the
free-tier wire); synonymy limit flagged honestly in Methodology.

Tabs:
  1. Search       — natural-language query → top-5 citations with highlighted terms
  2. Explore      — browse books / chapters / chunks directly
  3. Vocabulary   — top terms per book, chunk length distribution
  4. Methodology  — 5 decisions + "what a staff engineer would flag"
  5. About
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"

BRAND_PRIMARY = "#9B59B6"  # GenAI purple
BRAND_YELLOW = "#FFC72C"
BRAND_GREEN = "#50C878"
BRAND_BLUE = "#4A90E2"
BRAND_RED = "#E63946"
BRAND_INK = "#0E1117"
BRAND_INK2 = "#171B22"
BRAND_INK3 = "#232833"
BRAND_FG = "#E7E9EC"
BRAND_FG_DIM = "#9AA3B2"

sys.path.insert(0, str(HERE / "src"))

st.set_page_config(
    page_title="DocRanger — TF-IDF RAG demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================= bootstrap

@st.cache_resource(show_spinner="First-time setup: chunking + TF-IDF (~5s)...")
def bootstrap() -> None:
    required = [
        CHECKPOINTS / "chunks.parquet",
        CHECKPOINTS / "vectorizer.pkl",
        CHECKPOINTS / "tfidf_matrix.npz",
    ]
    if all(p.exists() for p in required):
        return
    CHECKPOINTS.mkdir(exist_ok=True)
    import phase1_chunk, phase2_embed
    phase1_chunk.main()
    phase2_embed.main()


bootstrap()


# ======================================================= loaders + retrieval

from phase3_retrieve import retrieve, compose, Hit  # noqa: E402


@st.cache_data
def load_chunks() -> pd.DataFrame:
    return pd.read_parquet(CHECKPOINTS / "chunks.parquet")


@st.cache_data
def vocab_stats() -> dict:
    import joblib
    from scipy import sparse
    # joblib.load deserializes pickle. Safe here because vectorizer.pkl is produced
    # by src/phase2_embed.py in this repo's own build.
    vec = joblib.load(CHECKPOINTS / "vectorizer.pkl")
    mat = sparse.load_npz(CHECKPOINTS / "tfidf_matrix.npz")
    terms = vec.get_feature_names_out()
    return {"n_terms": int(len(terms)), "shape": mat.shape, "nnz": int(mat.nnz)}


@st.cache_data
def top_terms_per_book(chunks: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """For each book, surface its top-k TF-IDF terms vs other books."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Fit a per-book-level TF-IDF (one doc per book) to find distinctive terms.
    per_book = chunks.groupby("book")["text"].apply(" ".join).reset_index()
    v = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1, max_df=0.9)
    X = v.fit_transform(per_book["text"])
    terms = v.get_feature_names_out()
    rows = []
    for i, book in enumerate(per_book["book"]):
        row = X[i].toarray().ravel()
        top = row.argsort()[-k:][::-1]
        for rank, idx in enumerate(top, 1):
            rows.append({"book": book, "rank": rank, "term": terms[idx], "weight": float(row[idx])})
    return pd.DataFrame(rows)


# ======================================================= Global CSS

st.markdown(
    f"""
    <style>
    #MainMenu, footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}
    .kpi-card {{
        background: linear-gradient(135deg, {BRAND_INK2} 0%, {BRAND_INK3} 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid var(--accent, {BRAND_PRIMARY});
        border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        height: 100%;
    }}
    .kpi-card .kpi-label {{color: {BRAND_FG_DIM}; font-size: 12px; font-weight: 500;
                          text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;}}
    .kpi-card .kpi-value {{color: {BRAND_FG}; font-size: clamp(16px, 1.9vw, 30px);
                          font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums;
                          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
    .kpi-card .kpi-delta {{display: inline-block; margin-top: 6px; color: {BRAND_FG_DIM};
                          font-size: 12px;}}
    .kpi-icon {{float: right; font-size: 20px; opacity: 0.35; margin-left: 4px;}}
    @media (max-width: 1100px) {{ .kpi-icon {{display: none;}} }}
    .insight {{background: {BRAND_INK2}; border: 1px solid rgba(155,89,182,0.22);
              border-radius: 10px; padding: 10px 14px; font-size: 13px;
              color: {BRAND_FG}; line-height: 1.45;}}
    .insight .insight-tag {{color: {BRAND_PRIMARY}; font-weight: 600; font-size: 11px;
                           text-transform: uppercase; letter-spacing: 0.07em; margin-right: 6px;}}
    .hero {{background: radial-gradient(circle at top left, rgba(155,89,182,0.14) 0%, rgba(14,17,23,0) 55%);
           padding: 10px 0 16px; margin-bottom: 10px;}}
    .hero h1 {{font-size: 36px !important; font-weight: 800 !important; margin-bottom: 4px !important;}}
    .hero .tagline {{color: {BRAND_FG_DIM}; font-size: 15px;}}
    section[data-testid="stSidebar"] {{background: {BRAND_INK};}}
    .citation {{background: {BRAND_INK2}; border-left: 3px solid {BRAND_PRIMARY};
               border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;}}
    .citation .cite-meta {{color: {BRAND_PRIMARY}; font-weight: 600; font-size: 12px;
                          text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;}}
    .citation .cite-text {{color: {BRAND_FG}; font-size: 14px; line-height: 1.55;}}
    .hl {{background: rgba(155,89,182,0.3); padding: 1px 3px; border-radius: 3px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, delta: str = "", icon: str = "", accent: str = BRAND_PRIMARY):
    st.markdown(
        f"""<div class="kpi-card" style="--accent:{accent};">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def highlight(text: str, terms: list[str]) -> str:
    """Wrap each term (case-insensitive, word boundary) in <span class='hl'>."""
    if not terms:
        return text
    pattern = r"\b(" + "|".join(re.escape(t) for t in terms if t.strip()) + r")\b"
    return re.sub(pattern, r'<span class="hl">\1</span>', text, flags=re.IGNORECASE)


# ======================================================= Load

chunks = load_chunks()
stats = vocab_stats()
books = sorted(chunks["book"].unique())


# ======================================================= Sidebar

with st.sidebar:
    st.markdown("### 📚 DocRanger")
    st.caption("Bits to Builds · GenAI capstone")
    st.divider()
    st.markdown("##### Data")
    st.caption(f"**{len(chunks)}** chunks")
    st.caption(f"**{len(books)}** books")
    st.caption(f"**{stats['n_terms']:,}** vocabulary terms")
    st.caption(f"**{stats['nnz']:,}** non-zero TF-IDF cells")

    st.markdown("##### Downloads")
    st.download_button("⬇ Download corpus (CSV)", chunks.to_csv(index=False).encode(),
                       "docranger_corpus.csv", "text/csv", use_container_width=True)

    st.divider()
    st.markdown("##### Links")
    st.markdown("[💻 Source on GitHub](https://github.com/PJsAcademy/docranger)")
    st.markdown("[📚 Bits to Builds](https://bitstobuilds.com)")


# ======================================================= Hero + KPIs

st.markdown(
    """<div class="hero">
      <h1>📚 DocRanger</h1>
      <div class="tagline">TF-IDF RAG over a synthesised 6-book corpus — cite-your-sources, zero-LLM, free-tier friendly · GenAI capstone of
      <a href="https://bitstobuilds.com" style="color:#FFC72C;">Bits to Builds</a></div>
    </div>""",
    unsafe_allow_html=True,
)

n_chunks = len(chunks)
n_books = len(books)
avg_wc = float(chunks["word_count"].mean())
n_terms = stats["n_terms"]
density = 100 * stats["nnz"] / (stats["shape"][0] * stats["shape"][1])

k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("Chunks", f"{n_chunks:,}", f"{n_books} books × ~{n_chunks//n_books}", "📄")
with k2: kpi_card("Books", f"{n_books}", "distinct corpora", "📖", BRAND_YELLOW)
with k3: kpi_card("Vocabulary", f"{n_terms:,}", "unique TF-IDF terms", "🔤", BRAND_BLUE)
with k4: kpi_card("Avg chunk", f"{avg_wc:.0f} wd", "words per chunk", "📏", BRAND_GREEN)
with k5: kpi_card("Index density", f"{density:.2f}%", "non-zero TF-IDF cells", "🧮", BRAND_PRIMARY)


# Insight pills
longest_book = chunks.groupby("book")["word_count"].sum().idxmax()
longest_chunk = chunks.loc[chunks["word_count"].idxmax()]
shortest_chunk = chunks.loc[chunks["word_count"].idxmin()]

st.markdown("")
i1, i2, i3, i4 = st.columns(4)
pills = [
    ("Longest book", f"<b>{longest_book}</b> has the most words."),
    ("Longest chunk", f"<b>{int(longest_chunk['word_count'])} words</b> ({longest_chunk['book']}, ch{longest_chunk['chapter']})."),
    ("Shortest chunk", f"<b>{int(shortest_chunk['word_count'])} words</b> ({shortest_chunk['book']}, ch{shortest_chunk['chapter']})."),
    ("Retrieval cost", f"<b>TF-IDF cosine</b> — no neural embeddings, ~5 MB on disk."),
]
for col, (tag, body) in zip([i1, i2, i3, i4], pills):
    with col:
        st.markdown(f'<div class="insight"><span class="insight-tag">{tag}</span>{body}</div>',
                    unsafe_allow_html=True)

st.markdown("")
st.divider()

tab_search, tab_explore, tab_vocab, tab_method, tab_about = st.tabs(
    ["🔎 Search", "📖 Explore", "🔤 Vocabulary", "🛠 Methodology", "ℹ️ About"]
)


# ------- Search tab -------
with tab_search:
    st.subheader("Natural-language search over the corpus")
    st.caption("TF-IDF cosine ranks the top passages across all 6 books. Matching terms highlighted. "
               "Out-of-domain queries return 0 hits (safety by default — no hallucination).")

    if "search_q" not in st.session_state:
        st.session_state.search_q = "sea storm lighthouse"

    examples = ["sea storm lighthouse", "garden flowers blooming",
                "clockmaker brass gear", "mountain glacier climbing",
                "stars telescope observation", "merchant spice silk"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.search_q = ex

    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Your question", key="search_q")
    k = c2.slider("Top-k", 1, 15, 5)

    if q:
        hits = retrieve(q, k=k)
        if not hits:
            st.warning(f"No passages matched **{q!r}** above the relevance threshold. "
                       "Try more specific terms.")
        else:
            st.success(f"Found **{len(hits)}** passages. Top score **{hits[0].score:.3f}**.")
            query_terms = re.findall(r"\w+", q.lower())
            for h in hits:
                st.markdown(
                    f"""<div class="citation">
                      <div class="cite-meta">[{h.rank}] {h.book} · Chapter {h.chapter}, passage {h.position} · score {h.score:.3f}</div>
                      <div class="cite-text">{highlight(h.text, query_terms)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Download the composed response
            composed = compose(q, hits)
            st.download_button(
                "⬇ Download composed answer (text)",
                composed.encode(), "docranger_answer.txt", "text/plain",
                use_container_width=False,
            )


# ------- Explore tab -------
with tab_explore:
    st.subheader("Browse the corpus directly")
    c1, c2 = st.columns([1, 2])
    with c1:
        pick_book = st.selectbox("Book", books, index=0)
        book_chunks = chunks[chunks["book"] == pick_book]
        pick_chapter = st.selectbox(
            "Chapter", sorted(book_chunks["chapter"].unique()), index=0,
        )
        chapter_chunks = book_chunks[book_chunks["chapter"] == pick_chapter]
    with c2:
        st.markdown(f"**{pick_book}** · Chapter {pick_chapter} · {len(chapter_chunks)} passages")
        for _, row in chapter_chunks.iterrows():
            st.markdown(
                f"""<div class="citation">
                  <div class="cite-meta">Passage {int(row['position'])} · chunk_id {int(row['chunk_id'])} · {int(row['word_count'])} words</div>
                  <div class="cite-text">{row['text']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ------- Vocabulary tab -------
with tab_vocab:
    st.subheader("Top distinctive terms per book")
    st.caption("Per-book TF-IDF computed with each book as one document, so these terms are what *differentiates* each book — not just its most frequent words.")
    top_terms = top_terms_per_book(chunks, k=10)

    pick = st.selectbox("Pick a book", books, index=0, key="vocab_book")
    this = top_terms[top_terms["book"] == pick].sort_values("rank")
    bar = (alt.Chart(this).mark_bar(cornerRadius=4).encode(
               x=alt.X("weight:Q", title="TF-IDF weight (book-level)"),
               y=alt.Y("term:N", sort="-x", title=None),
               color=alt.Color("weight:Q", scale=alt.Scale(scheme="purples"), legend=None),
               tooltip=["term", alt.Tooltip("weight:Q", format=".3f"), "rank"],
           ).properties(height=320))
    st.altair_chart(bar, use_container_width=True)

    st.divider()
    st.subheader("Chunk length distribution")
    hist = (alt.Chart(chunks).mark_bar(cornerRadius=4, color=BRAND_PRIMARY).encode(
                x=alt.X("word_count:Q", bin=alt.Bin(maxbins=20), title="Words per chunk"),
                y=alt.Y("count()", title="Chunks"),
                tooltip=["count()"],
            ).properties(height=240))
    st.altair_chart(hist, use_container_width=True)


# ------- Methodology tab -------
with tab_method:
    st.markdown(
        """
        ## Methodology — decisions, tradeoffs, honest shortcuts

        ---
        ### Decision 1 — Why TF-IDF instead of sentence-transformers?

        **Chose:** sklearn `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` over the chunks.

        **Why:** Streamlit Cloud free tier has 1 GB RAM / 1 GB storage. Torch alone is ~500 MB
        installed; a MiniLM checkpoint adds ~90 MB; cold-start downloads another ~400 MB
        the first time. For a demo, that's a 2-5 minute cold-start before the UI appears.
        TF-IDF on 180 chunks fits in **~5 MB** and runs sub-millisecond. On a well-themed
        corpus it recovers the right book nearly every time (see the search examples).

        **What I'd change with 10× the time:** a quantised MiniLM ONNX model (~15 MB)
        loaded via `onnxruntime`. Semantic retrieval ("investigator" matches "detective")
        at a tenth the cost of torch. The retrieval API stays the same.

        ---
        ### Decision 2 — Why no LLM response synthesis?

        **Chose:** Template-based `compose()` returns the top passages verbatim with
        citations. No prose synthesis.

        **Why:** An LLM is where RAG goes wrong in two ways: hallucination (claims not in
        the retrieved context) and dependency (API key, rate limits, cost). The honest
        demo is: here's deterministic retrieval + here's where you'd plug an LLM, with
        the system prompt that forbids ungrounded claims. The user ships quotes they can
        verify against the source.

        **What I'd change with 10× the time:** `claude-sonnet-5-5` or `claude-haiku-5-5`
        called with the top-5 passages as system prompt + "answer only from these passages
        and cite" rule + an output validator that re-fetches each cited claim and checks
        it appears verbatim. See `claude-api` skill for the Anthropic SDK pattern.

        ---
        ### Decision 3 — Why synthesise instead of Project Gutenberg classics?

        **Chose:** 6 thematically-distinct faux-books × 10 chapters × 3 passages
        (seed=42, 180 chunks total).

        **Why (honest):** Real Project Gutenberg texts are 100-500 KB each; six of them
        easily inflate the repo. The RAG *mechanism* is what the capstone teaches —
        not whether the corpus is Dickens or Tolstoy. The synth corpus has crisp theme
        separation that makes relevance obvious; a real corpus adds noise that obscures
        what the retriever is doing.

        **What I'd change with 10× the time:** add a GitHub Action that pulls 100 top
        Project Gutenberg books, chunks them, re-fits TF-IDF, and commits the artifacts
        nightly. UI stays identical; corpus just gets bigger and richer.

        ---
        ### Decision 4 — Why cosine threshold 1e-6 for "no hit", not top-k always?

        **Chose:** Filter out chunks with cosine similarity ≤ 1e-6 **before** picking
        top-k.

        **Why:** The default "always return top-k" behavior lies to users: query
        "pineapple elephant rocketship" over a lighthouse-and-flowers corpus returns 5
        chunks ranked by tiny coincidental overlaps. That's exactly the hallucination
        pattern users should see as "no match", not "here are some irrelevant guesses".
        Reporting 0 hits is the honest answer; it's also what the LLM step needs to
        decide "I shouldn't answer this".

        **What I'd change with 10× the time:** calibrate the threshold per-corpus with a
        held-out set of known-off-topic queries. 1e-6 is a safe floor; a learned
        threshold could be tighter.

        ---
        ### Decision 5 — Why word-boundary highlighting instead of substring?

        **Chose:** Regex `\\b<term>\\b` for highlighting matched query terms.

        **Why:** Substring highlighting over "glacier" would also light up "glacial" and
        "glaciers" — plausibly useful but easy to get subtly wrong (e.g. "ice" also
        highlights "service"). Word-boundary matching is predictable and safe. Users
        trust highlights that match literally.

        **What I'd change with 10× the time:** stem query terms the same way `sklearn`
        does and highlight stems, so "flowers" also lights up "flower" and "flowering"
        — matching how the retriever itself scores.

        ---

        ## What a staff engineer would flag that I left in

        - **TF-IDF misses synonymy and paraphrase.** A query for "investigator" won't
          match a chunk about "detective". Flagged explicitly; the ONNX MiniLM path fixes it.
        - **No conversation state.** Each query is independent; the UI doesn't remember
          previous turns. A real "chat with your docs" product would thread the dialogue.
        - **Chunks are fixed-size by generation, not meaning-aware.** Real documents need
          sentence-boundary chunking + overlap. `langchain.text_splitter.RecursiveCharacterTextSplitter`
          is the standard pattern.
        - **No source metadata hierarchy.** Books have chapters but no section/subsection
          structure. Real citations include page numbers, line ranges, publication IDs.
        """
    )


# ------- About tab -------
with tab_about:
    st.markdown(
        f"""
        ## About

        **DocRanger** is the GenAI capstone of the [Bits to Builds](https://bitstobuilds.com)
        course. A deterministic, LLM-free RAG demo built on the smallest possible retrieval
        stack (sklearn `TfidfVectorizer` + cosine similarity).

        | Phase | Deliverable |
        |-------|-------------|
        | 1. Chunk    | Synthesise 6 themed books × 10 ch × 3 passages = {n_chunks} chunks, {chunks['word_count'].sum():,} words |
        | 2. Index    | sklearn TF-IDF (unigrams + bigrams, sublinear_tf), ~5 MB total |
        | 3. Retrieve | Cosine top-k with 1e-6 relevance threshold + cite-your-sources template |

        ## Corpus

        6 synthesised themed books: *The Lighthouse Keeper*, *A Study of Flowers*,
        *The Clockmaker's Apprentice*, *Reports from the Expedition*, *Letters from
        the Observatory*, *Notes on the Market*. All synthetic → zero copyright
        questions. The capstone teaches the RAG mechanism, not any specific text.

        ## Honest limits

        - **No semantic retrieval.** TF-IDF only. Synonyms miss. Methodology tab has the
          ONNX MiniLM upgrade path.
        - **No LLM synthesis.** Response is top passages verbatim with citations — safer
          but can't paraphrase.
        - **Synth corpus.** The books are plausible-sounding but not real literature. All
          numbers on this deploy illustrate the pipeline, not literary scholarship.

        ## Source
        [github.com/PJsAcademy/docranger](https://github.com/PJsAcademy/docranger)
        """
    )
