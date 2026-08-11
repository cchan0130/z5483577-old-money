"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md), cap the sample
at 2023-12-31, and drop only EXACT duplicate rows. Real outliers are kept and
documented, never deleted (that screen lives in Station 2 + quality.py). Do not
commit data files.
"""
import pandas as pd

from src import data_access

# Coverage cap: the datasets end in 2023; ignore anything after this calendar day.
CAP_DATE = pd.Timestamp("2023-12-31")


def _utc_calendar_date(s: pd.Series) -> pd.Series:
    """tz-aware UTC timestamps -> tz-naive calendar date (midnight, dtype datetime64).

    The news `date` is `datetime64[us, UTC]`; prices are tz-naive. Convert to the
    UTC wall-clock, drop the tz, then normalise to midnight so the result is a
    plain calendar date that lines up with the (tz-naive) price dates.
    """
    return s.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()


def load_clean_equities(drop_dupes: bool = True) -> pd.DataFrame:
    """Load equity prices, cap at 2023-12-31, drop exact (ticker, date) duplicates.

    Prices are unique by ticker+date, so an exact (ticker, date) duplicate is a
    data error; keep the first. Pass ``drop_dupes=False`` to get the capped frame
    *before* de-duplication (used by the duplicate audit).
    """
    df = data_access.load_equity_prices()
    df = df[df["date"] <= CAP_DATE].copy()
    if not drop_dupes:
        return df.reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first").reset_index(drop=True)
    print(f"[equities] dropped {n_before - len(df)} exact (ticker,date) duplicate rows")
    return df


def load_clean_crypto(drop_dupes: bool = True) -> pd.DataFrame:
    """Load crypto prices, cap at 2023-12-31, drop exact (ticker, date) duplicates.

    The cap removes the 10 stray 2024-01-01 rows. Crypto runs a 7-day calendar but
    is still unique by ticker+date. Pass ``drop_dupes=False`` for the capped frame
    before de-duplication.
    """
    df = data_access.load_crypto_prices()
    df = df[df["date"] <= CAP_DATE].copy()
    if not drop_dupes:
        return df.reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first").reset_index(drop=True)
    print(f"[crypto] dropped {n_before - len(df)} exact (ticker,date) duplicate rows")
    return df


def load_clean_news(drop_dupes: bool = True) -> pd.DataFrame:
    """Load headlines, cap at 2023-12-31, drop exact (ticker, date, title) duplicates.

    Many headlines per ticker-date is normal, so de-dup on the full triple, not on
    ticker+date. The title text is left RAW (no lowercasing or stripping) because
    VADER needs it verbatim in Part B. A tz-naive calendar-date column
    ``date_naive`` is added for later merges; the original tz-aware ``date`` is kept.
    Pass ``drop_dupes=False`` for the capped frame before de-duplication.
    """
    df = data_access.load_news_headlines().copy()
    df["date_naive"] = _utc_calendar_date(df["date"])
    df = df[df["date_naive"] <= CAP_DATE].copy()
    if not drop_dupes:
        return df.reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first").reset_index(drop=True)
    print(f"[news] dropped {n_before - len(df)} exact (ticker,date,title) duplicate headlines")
    return df
