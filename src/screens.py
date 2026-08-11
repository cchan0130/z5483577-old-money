"""Station 2 - extreme-value (outlier) screen on daily returns.

These are REAL market events (crashes, single-name shocks), not data errors, so
the screen flags and documents them but never drops them (see CLAUDE.md). Two
rules run side by side: a fixed absolute threshold and a robust modified z-score
computed within each ticker.
"""
import numpy as np
import pandas as pd

FIXED_THRESHOLD = 0.25   # |ret| > 25% in a single day
MAD_Z_THRESHOLD = 5.0    # |modified z| > 5
_MAD_SCALE = 0.6745      # 0.6745 = Phi^{-1}(0.75); scales MAD to a std estimate

# Simple, factual cause windows for the writeup. Extend as needed.
_CAUSE_WINDOWS = [
    ("COVID crash (Mar-2020)", "2020-02-20", "2020-04-30"),
]


def _modified_zscore(r: pd.Series) -> pd.Series:
    """Robust modified z-score within one ticker: 0.6745*(r - median)/MAD.

    MAD is the median absolute deviation. If MAD is 0 (degenerate) the score is 0
    so a flat series is never flagged.
    """
    median = r.median()
    mad = (r - median).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(0.0, index=r.index)
    return _MAD_SCALE * (r - median) / mad


def outlier_screen(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Flag extreme daily returns by a fixed threshold and a within-ticker MAD score.

    Returns only the flagged rows: date, ticker, asset_class, ret, flag_fixed,
    flag_mad, sorted by absolute return (largest first). A row is returned if
    either rule fires. Nothing is dropped from the underlying data.
    """
    df = returns_long.copy()
    df["flag_fixed"] = df["ret"].abs() > FIXED_THRESHOLD
    df["mod_z"] = df.groupby("ticker", sort=False)["ret"].transform(_modified_zscore)
    df["flag_mad"] = df["mod_z"].abs() > MAD_Z_THRESHOLD

    flagged = df[df["flag_fixed"] | df["flag_mad"]].copy()
    flagged = flagged.reindex(
        flagged["ret"].abs().sort_values(ascending=False).index
    )
    cols = ["date", "ticker", "asset_class", "ret", "flag_fixed", "flag_mad"]
    return flagged[cols].reset_index(drop=True)


def tag_cause_window(flagged: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Tag the largest flagged moves with a plausible cause window, for the writeup.

    Keeps it factual: a date inside a known window (e.g. the Mar-2020 COVID crash)
    gets that label, otherwise "single-name / other". Only the ``top_n`` most
    extreme rows are returned, with columns date, ticker, asset_class, ret, cause.
    """
    top = flagged.head(top_n).copy()
    dates = pd.to_datetime(top["date"])
    cause = pd.Series("single-name / other", index=top.index)
    for label, start, end in _CAUSE_WINDOWS:
        in_window = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        cause = cause.mask(in_window, label)
    top["cause"] = cause
    return top[["date", "ticker", "asset_class", "ret", "cause"]].reset_index(drop=True)
