"""Station 3 - your sentiment model and index from news headlines.

BUILD step only: this module imports nltk and scores headlines with VADER. The
deployed app must NEVER import this - it loads the precomputed
results/data/sector_sentiment_index.csv instead (see CLAUDE.md).

VADER is rule-based and reads casing, punctuation, emphasis and negation, so the
headline text is scored RAW - no lowercasing, punctuation stripping, or negation
removal (etl.load_clean_news already keeps titles verbatim). Sentiment work is
equity-only. The one-trading-day lag is applied later, in the fusion, not here.
"""
from __future__ import annotations

import pandas as pd

from src import features


def _ensure_vader():
    """Download the VADER lexicon once if missing, then return an analyzer.

    Guarded so a repeat run never re-downloads. This is the only network call in
    the sentiment build; it must not run inside the deployed app. On macOS the
    default Python SSL context often lacks a CA bundle ("certificate verify
    failed"); if the first download fails we retry with certifi's CA bundle (the
    one requests already uses) before giving up.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    def _have_lexicon() -> bool:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
            return True
        except LookupError:
            return False

    if not _have_lexicon():
        nltk.download("vader_lexicon")
        if not _have_lexicon():
            import ssl
            import certifi
            ssl._create_default_https_context = (
                lambda *a, **k: ssl.create_default_context(cafile=certifi.where()))
            nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


def score_headlines(panel: pd.DataFrame, trading_calendar=None) -> pd.DataFrame:
    """Score raw headlines with VADER and aggregate to a per-(ticker, day) score.

    ``panel`` is the cleaned news frame from ``etl.load_clean_news`` (raw ``title``
    and a tz-naive ``date_naive``). Steps:

    1. Align each headline to its equity trading day with
       ``features.assemble_headline_panel`` (same-day if it trades, else the next
       trading day). Pass ``trading_calendar`` (from
       ``quality.equity_trading_calendar``); if omitted it is derived from the
       equity prices.
    2. Score each RAW headline's VADER ``compound`` in [-1, 1].
    3. Aggregate to ``sent`` = mean compound of a ticker's headlines on that
       trading day.

    The share of headlines whose compound is exactly 0 (VADER-neutral) is reported
    at ``result.attrs['neutral_share']`` so the neutral-heavy caveat can be flagged.

    Returns tidy: trading_day, ticker, sector, sent.
    """
    if trading_calendar is None:
        from src import data_access, quality
        trading_calendar = quality.equity_trading_calendar(data_access.load_equity_prices())

    daily_panel = features.assemble_headline_panel(panel, trading_calendar)

    analyzer = _ensure_vader()
    # Score raw titles verbatim (no casing/punctuation/negation stripping).
    compound = daily_panel["title"].map(lambda t: analyzer.polarity_scores(t)["compound"])
    neutral_share = float((compound == 0.0).mean())

    scored = daily_panel.assign(compound=compound)
    scores = (scored.groupby(["trading_day", "ticker", "sector"], sort=True)["compound"]
                    .mean()
                    .reset_index()
                    .rename(columns={"compound": "sent"}))
    scores.attrs["neutral_share"] = neutral_share
    scores.attrs["n_headlines_scored"] = int(len(daily_panel))
    return scores


def daily_ticker_sentiment(scores: pd.DataFrame, trading_calendar=None) -> pd.DataFrame:
    """Per-ticker daily sentiment on the full trading calendar (forward-filled).

    No-headline handling (forward-fill): a ticker's daily score is carried forward
    across the full trading calendar until fresh news arrives, and is a neutral 0
    before that ticker's first headline. News sentiment persists until it is
    updated; forward-filling avoids a jumpy series that would otherwise swing with
    whichever tickers happen to have news on a given day (a composition artefact,
    not a sentiment move).

    Returns wide: index = trading_day, columns = ticker, defined on every trading
    day. The fusion lags this (uses the value dated strictly before a rebalance).
    """
    if trading_calendar is None:
        from src import data_access, quality
        trading_calendar = quality.equity_trading_calendar(data_access.load_equity_prices())
    cal = pd.DatetimeIndex(trading_calendar).sort_values()

    per_ticker = (scores.pivot(index="trading_day", columns="ticker", values="sent")
                        .reindex(cal)
                        .ffill()
                        .fillna(0.0))
    per_ticker.index.name = "trading_day"
    return per_ticker


def sector_sentiment_index(scores: pd.DataFrame, trading_calendar=None) -> pd.DataFrame:
    """Equal-weight sector sentiment index on every trading day.

    Forward-fills each ticker's score to the full calendar (see
    ``daily_ticker_sentiment`` for the no-headline choice) and takes the sector
    index for each trading day as the equal-weight mean of that sector's tickers'
    filled scores.

    Returns wide: index = trading_day, columns = the 10 sectors, defined on every
    trading day in the calendar.
    """
    ticker_sector = (scores[["ticker", "sector"]].drop_duplicates()
                     .set_index("ticker")["sector"])
    per_ticker = daily_ticker_sentiment(scores, trading_calendar)

    # Equal-weight mean across each sector's tickers, per trading day.
    sector_cols = {}
    for sector, group in ticker_sector.groupby(ticker_sector):
        sector_cols[sector] = per_ticker[group.index].mean(axis=1)
    index = pd.DataFrame(sector_cols).sort_index(axis=1)
    index.index.name = "trading_day"
    return index
