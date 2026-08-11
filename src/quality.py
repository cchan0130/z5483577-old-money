"""Station 1 - data integrity audits.

Integrity is a distinct concern from loading, so it lives here rather than in
etl.py. These functions summarise what the cleaned datasets look like: the
reference trading calendar, a duplicate audit, a per-ticker missing-date audit,
an OHLC consistency check, and a single tidy integrity summary. Part A confirms
consistency and KEEPS rows; nothing here deletes data.

The returns-based extreme-value (outlier) screen needs daily returns (a Station 2
feature), so that check is run in the next task and its row is appended to the
integrity summary there.
"""
import pandas as pd

_PRICE_KEY = ["ticker", "date"]
_NEWS_KEY = ["ticker", "date", "title"]


def equity_trading_calendar(equities: pd.DataFrame) -> pd.DatetimeIndex:
    """Reference trading calendar: sorted union of every equity date.

    Later merges align crypto and news to this calendar, so it is the single
    source of truth for "which days the market traded".
    """
    return pd.DatetimeIndex(pd.unique(equities["date"])).sort_values()


def duplicate_audit(equities: pd.DataFrame, crypto: pd.DataFrame,
                    news: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: raw rows, exact duplicates, dedup key, rows after.

    Pass the CAPPED, pre-dedup frames (``etl.load_clean_*(drop_dupes=False)``) so
    ``n_after`` matches the final cleaned row count. Prices dedup on (ticker, date);
    news on (ticker, date, title) because many headlines share a ticker-date.
    """
    specs = [
        ("equities", equities, _PRICE_KEY),
        ("crypto", crypto, _PRICE_KEY),
        ("news", news, _NEWS_KEY),
    ]
    rows = []
    for name, df, key in specs:
        n_raw = len(df)
        n_dupes = int(df.duplicated(subset=key, keep="first").sum())
        rows.append({
            "dataset": name,
            "n_rows_raw": n_raw,
            "n_exact_dupes": n_dupes,
            "dedup_key": "+".join(key),
            "n_after": n_raw - n_dupes,
        })
    return pd.DataFrame(rows)


def missing_date_audit(equities: pd.DataFrame, crypto: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker coverage: observed vs expected days within each ticker's span.

    Equities expect the trading-calendar days that fall inside the ticker's
    [first, last]. Crypto expects every calendar day (7-day) in its span. A
    positive ``n_missing`` means gaps a merge would have to handle.
    """
    calendar = equity_trading_calendar(equities)
    rows = []

    for ticker, g in equities.groupby("ticker", sort=True):
        first, last = g["date"].min(), g["date"].max()
        n_obs = g["date"].nunique()
        n_expected = int(((calendar >= first) & (calendar <= last)).sum())
        rows.append({
            "ticker": ticker, "asset_class": "equity",
            "first_date": first, "last_date": last,
            "n_obs": n_obs, "n_expected": n_expected,
            "n_missing": n_expected - n_obs,
        })

    for ticker, g in crypto.groupby("ticker", sort=True):
        first, last = g["date"].min(), g["date"].max()
        n_obs = g["date"].nunique()
        n_expected = int((last - first).days) + 1  # inclusive 7-day span
        rows.append({
            "ticker": ticker, "asset_class": "crypto",
            "first_date": first, "last_date": last,
            "n_obs": n_obs, "n_expected": n_expected,
            "n_missing": n_expected - n_obs,
        })

    cols = ["ticker", "asset_class", "first_date", "last_date",
            "n_obs", "n_expected", "n_missing"]
    return pd.DataFrame(rows, columns=cols)


def ohlc_consistency(prices: pd.DataFrame, label: str, sample_n: int = 5) -> pd.DataFrame:
    """Bar-level sanity check on a price panel; returns a one-row summary.

    Flags bars where low<=open<=high or low<=close<=high is violated, plus
    non-negative volume and positive adjClose. A sample of offending rows is
    attached at ``result.attrs['offenders']`` (Part A keeps every row, so this is
    for documentation, not deletion).
    """
    low, high = prices["low"], prices["high"]
    op, cl = prices["open"], prices["close"]

    ohlc_ok = (low <= op) & (op <= high) & (low <= cl) & (cl <= high)
    bad_ohlc = ~ohlc_ok
    bad_volume = prices["volume"] < 0
    bad_adjclose = prices["adjClose"] <= 0

    offenders = prices[bad_ohlc | bad_volume | bad_adjclose]
    summary = pd.DataFrame([{
        "dataset": label,
        "n_rows": len(prices),
        "n_bad_ohlc": int(bad_ohlc.sum()),
        "n_bad_volume": int(bad_volume.sum()),
        "n_bad_adjclose": int(bad_adjclose.sum()),
    }])
    summary.attrs["offenders"] = offenders.head(sample_n).copy()
    return summary


def build_integrity_summary(dup_audit: pd.DataFrame,
                            ohlc_summaries: list[pd.DataFrame],
                            cap_audit: pd.DataFrame) -> pd.DataFrame:
    """Fold the audits into one tidy table: check, dataset, n_flagged, resolution.

    Args:
        dup_audit: output of ``duplicate_audit``.
        ohlc_summaries: list of ``ohlc_consistency`` one-row summaries.
        cap_audit: columns ``dataset``, ``n_capped`` = rows removed by the
            2023-12-31 cap.

    The returns-based extreme-value screen is appended in the next task.
    """
    rows = []

    for _, r in cap_audit.iterrows():
        rows.append({
            "check": "date cap (>2023-12-31)",
            "dataset": r["dataset"],
            "n_flagged": int(r["n_capped"]),
            "resolution": "dropped (outside coverage)",
        })

    for _, r in dup_audit.iterrows():
        rows.append({
            "check": "exact duplicates",
            "dataset": r["dataset"],
            "n_flagged": int(r["n_exact_dupes"]),
            "resolution": f"dropped, keeping first on ({r['dedup_key']})",
        })

    for s in ohlc_summaries:
        r = s.iloc[0]
        n_flagged = int(r["n_bad_ohlc"] + r["n_bad_volume"] + r["n_bad_adjclose"])
        rows.append({
            "check": "OHLC / volume / adjClose consistency",
            "dataset": r["dataset"],
            "n_flagged": n_flagged,
            "resolution": "consistent; rows kept" if n_flagged == 0 else "flagged; rows kept for review",
        })

    return pd.DataFrame(rows, columns=["check", "dataset", "n_flagged", "resolution"])
