"""Phase 1 — DocRanger chunking.

Deterministic synthesised corpus of 6 themed books × 10 chapters × 3 paragraphs
= 180 chunks (~90 KB). Public domain by construction — no copyright. Each
book has a distinct theme and vocabulary so TF-IDF retrieval has something
to distinguish.

Writes checkpoints/chunks.parquet with: book, chapter, chunk_id, position,
text, word_count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)


# Themed vocabulary pools. Chunks concatenate ~40-80 words from each pool
# plus a per-chapter theme sentence. Deterministic from seed.
BOOK_THEMES = {
    "The Lighthouse Keeper": {
        "nouns": ["lighthouse", "lantern", "sea", "coast", "storm", "beacon", "gull",
                  "fog", "cliff", "tide", "watchman", "lamp", "night", "waves",
                  "rocks", "ship", "sailor", "horizon", "rain", "wind"],
        "verbs": ["flickered", "rolled", "crashed", "guided", "watched", "burned",
                  "turned", "signalled", "sheltered", "swept", "warned", "lit",
                  "faded", "returned", "sailed", "trembled"],
        "adjectives": ["dark", "silent", "distant", "cold", "wet", "lonely", "steady",
                       "bright", "weary", "ancient", "endless", "pale"],
        "theme_sentence": (
            "The lighthouse keeper trimmed the wick and watched the beam cross the "
            "water, counting the seconds between each sweep."
        ),
    },
    "A Study of Flowers": {
        "nouns": ["garden", "petal", "stem", "bloom", "roots", "sun", "rain", "dew",
                  "orchid", "rose", "fern", "soil", "butterfly", "greenhouse",
                  "lavender", "pollen", "bee", "shadow", "moss", "breeze"],
        "verbs": ["grew", "wilted", "blossomed", "turned", "unfolded", "lingered",
                  "pollinated", "shaded", "sprouted", "swayed", "leaned",
                  "scattered", "nurtured", "bloomed"],
        "adjectives": ["purple", "fragrant", "delicate", "wild", "patient", "soft",
                       "tender", "scented", "warm", "quiet", "verdant", "golden"],
        "theme_sentence": (
            "She bent over the orchid, considering how the petals curled against the "
            "morning light, and sketched quickly before the dew lifted."
        ),
    },
    "The Clockmaker's Apprentice": {
        "nouns": ["workshop", "clock", "gear", "pendulum", "spring", "escapement",
                  "oil", "brass", "winding", "mainspring", "chime", "glass",
                  "balance", "ratchet", "pinion", "tweezers", "tooth", "arbor",
                  "weight", "dial"],
        "verbs": ["calibrated", "wound", "ticked", "aligned", "set", "measured",
                  "dismantled", "polished", "adjusted", "turned", "assembled",
                  "regulated", "tested", "repaired", "sprung"],
        "adjectives": ["precise", "small", "tarnished", "intricate", "careful",
                       "steady", "methodical", "delicate", "exact", "old",
                       "patient", "brass"],
        "theme_sentence": (
            "The apprentice held the escapement up to the lamp and watched the "
            "balance wheel breathe, counting each tick against her pulse."
        ),
    },
    "Reports from the Expedition": {
        "nouns": ["glacier", "ridge", "camp", "compass", "map", "snow", "ice",
                  "rope", "crevasse", "altitude", "oxygen", "porter", "summit",
                  "tent", "radio", "frostbite", "boots", "crampons", "wind",
                  "ascent"],
        "verbs": ["climbed", "measured", "mapped", "traversed", "secured",
                  "radioed", "recorded", "sheltered", "descended", "camped",
                  "roped", "sighted", "broke", "pressed", "pitched"],
        "adjectives": ["frozen", "steep", "thin", "treacherous", "white", "howling",
                       "exhausted", "determined", "distant", "sharp", "icy",
                       "unmapped"],
        "theme_sentence": (
            "At the ridge the wind took the ends of the rope and we tied in short, "
            "watching for the slow blue cracks that mean a glacier is listening."
        ),
    },
    "Letters from the Observatory": {
        "nouns": ["telescope", "star", "nebula", "constellation", "planet", "orbit",
                  "satellite", "spectrum", "radiation", "pulsar", "galaxy", "comet",
                  "parallax", "redshift", "lens", "aperture", "declination",
                  "catalogue", "moon", "equator"],
        "verbs": ["observed", "catalogued", "measured", "resolved", "traced",
                  "photographed", "logged", "computed", "detected", "charted",
                  "integrated", "recorded", "aligned", "calibrated", "focused"],
        "adjectives": ["distant", "faint", "ancient", "spiral", "pale", "cold",
                       "deep", "stellar", "orbital", "silent", "clear", "measured"],
        "theme_sentence": (
            "At three in the morning the clouds parted and the comet appeared "
            "exactly where the catalogue had promised it, trailing a slow blue tail."
        ),
    },
    "Notes on the Market": {
        "nouns": ["ledger", "coin", "merchant", "stall", "spice", "silk", "cargo",
                  "dock", "ship", "warehouse", "trader", "bargain", "manifest",
                  "quay", "sack", "clerk", "harbour", "weight", "scale", "contract"],
        "verbs": ["bartered", "weighed", "haggled", "shipped", "unloaded",
                  "recorded", "paid", "signed", "measured", "stacked", "appraised",
                  "traded", "balanced", "stored", "delivered"],
        "adjectives": ["heavy", "fragrant", "busy", "worn", "shrewd", "fair",
                       "loud", "crowded", "foreign", "patient", "pungent", "quick"],
        "theme_sentence": (
            "The merchant turned the cardamom in her hand, smelled it twice, and "
            "named a price lower than the quayside gossip had led the clerk to expect."
        ),
    },
}


def _make_chunk(rng: np.random.Generator, theme: dict, book: str,
                chapter: int, position: int) -> str:
    """Build one ~60-word chunk from the book's vocabulary + theme sentence."""
    n_sentences = int(rng.integers(3, 5))
    sentences = [theme["theme_sentence"]]
    for _ in range(n_sentences - 1):
        length = int(rng.integers(6, 14))
        words = []
        for _ in range(length):
            pool = rng.choice(["nouns", "verbs", "adjectives", "nouns"], p=[0.4, 0.25, 0.2, 0.15])
            words.append(str(rng.choice(theme[pool])))
        sentence = " ".join(["The"] + words).capitalize() + "."
        sentences.append(sentence)
    return " ".join(sentences)


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    chunk_id = 0
    for book, theme in BOOK_THEMES.items():
        for chapter in range(1, 11):
            for position in range(1, 4):
                text = _make_chunk(rng, theme, book, chapter, position)
                rows.append({
                    "chunk_id": chunk_id,
                    "book": book,
                    "chapter": chapter,
                    "position": position,
                    "text": text,
                    "word_count": len(text.split()),
                })
                chunk_id += 1

    df = pd.DataFrame(rows)
    df.to_parquet(CHECKPOINTS / "chunks.parquet", index=False)
    print(f"[phase1] wrote {len(df):,} chunks from {df['book'].nunique()} books "
          f"({df['word_count'].sum():,} words total, "
          f"avg {df['word_count'].mean():.1f} words/chunk)")
    for book in BOOK_THEMES:
        n = (df["book"] == book).sum()
        print(f"  {book:35s} {n:>4} chunks")


if __name__ == "__main__":
    main()
