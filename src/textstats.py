"""Station 2 - descriptive text analytics on the headline panel (COUNTING ONLY).

Everything here counts: headlines over time, per ticker, per publisher, and word
membership in a sentiment vocabulary. Nothing here SCORES sentiment - the word
count uses vocabulary MEMBERSHIP only, never the VADER scores. Scoring (and
lagging the signal) is the Station 3 model, built in Part B.
"""
import re
from collections import Counter

import pandas as pd

# Token = run of letters/apostrophes; lowercased for MATCHING only (panel stays raw).
_TOKEN_RE = re.compile(r"[a-z']+")

# Fallback finance sentiment vocabulary, used when nltk/VADER is not installed.
_FALLBACK_VOCAB = {
    "gain", "gains", "gained", "loss", "losses", "up", "down", "surge", "surges",
    "plunge", "plunges", "rally", "rallies", "crash", "crashes", "beat", "beats",
    "miss", "misses", "cut", "cuts", "raise", "raises", "upgrade", "upgrades",
    "downgrade", "downgrades", "bullish", "bearish", "strong", "weak", "record",
    "high", "highs", "low", "lows", "fear", "fears", "risk", "risks", "boom",
    "bust", "soar", "soars", "tumble", "tumbles", "jump", "jumps", "fall",
    "falls", "rise", "rises", "drop", "drops", "profit", "profits", "warning",
    "warns", "fraud", "lawsuit", "win", "wins", "loses", "slump", "slumps",
    "rebound", "recovery", "growth", "decline", "declines", "outperform",
    "underperform", "positive", "negative", "optimistic", "pessimistic",
}


def _sentiment_vocab() -> tuple[set, str]:
    """Return (vocabulary word set, source label).

    Prefer the VADER lexicon KEY SET (membership only, scores ignored). Fall back
    to a small curated finance list when nltk is unavailable.
    """
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            sia = SentimentIntensityAnalyzer()
        return set(sia.lexicon.keys()), "vader_lexicon (membership only)"
    except Exception:
        return set(_FALLBACK_VOCAB), "fallback_finance_list (membership only)"


def articles_over_time(panel: pd.DataFrame) -> pd.DataFrame:
    """Headline counts per calendar month, overall and per sector.

    Tidy long: month (YYYY-MM), sector ("ALL" for the overall row), n_headlines.
    Months come from the original headline date (news_date).
    """
    p = panel.copy()
    p["month"] = pd.to_datetime(p["news_date"]).dt.to_period("M").astype(str)

    overall = p.groupby("month").size().reset_index(name="n_headlines")
    overall["sector"] = "ALL"
    by_sector = p.groupby(["month", "sector"]).size().reset_index(name="n_headlines")

    out = pd.concat(
        [overall[["month", "sector", "n_headlines"]], by_sector], ignore_index=True
    )
    return out.sort_values(["month", "sector"]).reset_index(drop=True)


def headlines_per_ticker(panel: pd.DataFrame) -> pd.DataFrame:
    """Total headlines per ticker, with its sector, busiest first."""
    out = (panel.groupby(["ticker", "sector"]).size()
           .reset_index(name="n_headlines"))
    return out.sort_values("n_headlines", ascending=False).reset_index(drop=True)


def publisher_coverage(headlines: pd.DataFrame) -> pd.DataFrame:
    """Headline counts by publisher, plus the share with a blank publisher.

    Takes the cleaned news frame (the panel drops the publisher column). A missing
    or whitespace-only publisher counts as blank; the overall blank share is at
    ``result.attrs['blank_share']``.
    """
    pub = headlines["publisher"].fillna("").astype(str).str.strip()
    is_blank = pub == ""
    labelled = pub.mask(is_blank, "(blank)")

    out = (labelled.value_counts().rename_axis("publisher")
           .reset_index(name="n_headlines"))
    out.attrs["blank_share"] = float(is_blank.mean())
    return out


def sentiment_word_counts(panel: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Count title tokens that belong to a sentiment vocabulary (membership only).

    Tokenises the RAW titles (lowercased for matching only), counts tokens whose
    lowercase form is IN the vocabulary, and returns the top_n most frequent such
    words. This is a vocabulary count, NOT sentiment scoring - the VADER scores
    are never used. Totals and the vocab source are at ``result.attrs``.
    """
    vocab, source = _sentiment_vocab()

    counts: Counter = Counter()
    total_tokens = 0
    for title in panel["title"].astype(str):
        for tok in _TOKEN_RE.findall(title.lower()):
            total_tokens += 1
            if tok in vocab:
                counts[tok] += 1

    out = pd.DataFrame(counts.most_common(top_n), columns=["word", "count"])
    out.attrs["vocab_source"] = source
    out.attrs["total_sentiment_tokens"] = int(sum(counts.values()))
    out.attrs["total_tokens"] = int(total_tokens)
    out.attrs["n_titles"] = int(len(panel))
    return out
