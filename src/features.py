"""Station 2 - your features: return features and headline text assembly.

Build your return (and optional risk) features here, and assemble the headlines
into a daily text panel. Part A is assembly only - scoring the text into sentiment
(and lagging the signal) is the Station 3 model, built in Part B, not here.

Returns are simple returns on adjClose, computed WITHIN each ticker's own panel
before any cross-calendar merge (see CLAUDE.md). Keep real outliers.
"""
import numpy as np
import pandas as pd

# Trading periods per year: equities ~252, crypto 365 (7-day calendar).
ANN_FACTOR = {"equity": 252, "crypto": 365}


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker, as a tidy long frame.

    Simple returns on adjClose, computed per ticker (sort by date, pct_change)
    BEFORE any merge, so a cross-calendar join never manufactures a spurious
    return. The first observation per ticker is NaN and is dropped. If ``prices``
    carries an ``asset_class`` column it is preserved.

    Returns columns: date, ticker, ret (+ asset_class when present).
    """
    keep = ["ticker", "date", price_col]
    has_class = "asset_class" in prices.columns
    if has_class:
        keep.append("asset_class")

    df = prices[keep].sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker", sort=False)[price_col].pct_change()
    df = df.dropna(subset=["ret"])  # drops the first obs per ticker

    out_cols = ["date", "ticker", "ret"] + (["asset_class"] if has_class else [])
    return df[out_cols].reset_index(drop=True)


def descriptive_stats(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Per asset_class return summary, daily and annualised.

    Reports count, mean, std (daily vol), min, max, skew, kurtosis, plus
    annualised mean (mean * factor) and annualised vol (std * sqrt(factor)).
    The annualisation factor differs by asset class (equity 252, crypto 365) and
    is reported in ``ann_factor`` so the scaling is explicit.
    """
    rows = []
    for asset_class, g in returns_long.groupby("asset_class", sort=True):
        r = g["ret"]
        factor = ANN_FACTOR[asset_class]
        mean, std = r.mean(), r.std()
        rows.append({
            "asset_class": asset_class,
            "count": int(r.count()),
            "mean": mean,
            "std": std,
            "min": r.min(),
            "max": r.max(),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),  # excess kurtosis (Fisher)
            "ann_factor": factor,
            "ann_mean": mean * factor,
            "ann_vol": std * np.sqrt(factor),
        })
    return pd.DataFrame(rows)


def assemble_headline_panel(headlines: pd.DataFrame,
                            trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Align each cleaned headline to its equity trading day (assembly only).

    A headline maps to the same day if its calendar date (`date_naive`) is a
    trading day, otherwise to the NEXT trading day. This is a forward
    ``merge_asof`` against the sorted equity calendar (first trading day >= the
    headline date). Headlines dated after the last trading day have no next
    trading day: they are flagged, dropped, and the count is reported at
    ``result.attrs['n_dropped_no_next_trading_day']`` (never silently lost).

    The title text is returned RAW - no lowercasing, stripping, punctuation
    removal, or stopword removal. Scoring and lagging the signal is the Station 3
    sentiment model, built in Part B, not here.

    Returns one row per surviving headline with columns:
    trading_day, ticker, sector, title (raw), news_date (original date_naive).
    """
    cal = pd.DataFrame(
        {"trading_day": pd.DatetimeIndex(trading_calendar).sort_values().as_unit("ns")}
    )
    left = headlines.copy()
    left["_key"] = pd.to_datetime(left["date_naive"]).astype("datetime64[ns]")
    left = left.sort_values("_key").reset_index(drop=True)

    merged = pd.merge_asof(
        left, cal, left_on="_key", right_on="trading_day", direction="forward"
    )
    n_dropped = int(merged["trading_day"].isna().sum())
    kept = merged.dropna(subset=["trading_day"]).copy()

    panel = kept.rename(columns={"date_naive": "news_date"})[
        ["trading_day", "ticker", "sector", "title", "news_date"]
    ].reset_index(drop=True)
    panel.attrs["n_dropped_no_next_trading_day"] = n_dropped
    return panel
