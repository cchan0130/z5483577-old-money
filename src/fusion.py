"""Station 3 (extension) - fuse sentiment into the funds.

Baseline tilt: nudge each equity fund's weights toward names with above-average
lagged sentiment, look-ahead safe, then test whether it helped. An honest
negative result, explained, is good work (a naive sentiment tilt often does NOT
beat the base fund).

Look-ahead safety: the sentiment used at rebalance ``t`` is dated strictly before
``t`` (>= 1 trading-day lag), so a weight decided at ``t`` never sees same-day or
future news. The tilt reuses the walk-forward structure of ``portfolios``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolios

ZERO_TOL = portfolios.ZERO_TOL


def apply_sentiment(base_weights: pd.Series, ticker_sentiment: pd.Series,
                    lam: float = 0.5) -> pd.Series:
    """Tilt long-only base weights toward high (lagged) sentiment names.

    Parameters
    ----------
    base_weights : Series
        Per-ticker weights from a ``portfolios`` weight function (sums to 1).
    ticker_sentiment : Series
        Each ticker's LAGGED sentiment as of the last trading day before the
        rebalance (forward-filled, shifted >= 1 day). Only the held names matter.
    lam : float
        Tilt strength (default 0.5). Larger ``lam`` tilts harder; a naive tilt may
        not help, so this is a modest baseline.

    Rule
    ----
    Standardise the lagged sentiment cross-sectionally across the HELD names
    (z-score), then set ``w_tilted proportional to base_weight * (1 + lam * z)``,
    clip at 0 (long-only), and renormalise to sum 1. Names the optimiser did not
    hold (base weight ~ 0) stay out. If fewer than two held names carry sentiment
    or the sentiment has no cross-sectional spread, the z-scores are 0 and the tilt
    is a no-op (the fund equals its base), which is the safe default.
    """
    held = base_weights[base_weights > ZERO_TOL]
    if held.empty:
        return base_weights.copy()

    s = ticker_sentiment.reindex(held.index)
    valid = s.dropna()
    sd = valid.std()
    if len(valid) < 2 or not np.isfinite(sd) or sd == 0:
        z = pd.Series(0.0, index=held.index)              # no usable spread -> no tilt
    else:
        z = ((s - valid.mean()) / sd).fillna(0.0)         # missing name -> neutral (z = 0)

    tilted = (held * (1.0 + lam * z)).clip(lower=0.0)      # long-only
    total = tilted.sum()
    if total <= 0:                                         # tilt wiped everything out -> fall back
        tilted = held.copy()
        total = tilted.sum()
    tilted = tilted / total
    return tilted.reindex(base_weights.index).fillna(0.0)  # non-held names stay 0


def fusion_backtest(returns_wide: pd.DataFrame, method: str,
                    ticker_sentiment_daily: pd.DataFrame,
                    periods_per_year: int = 252, lam: float = 0.5,
                    warmup: int = 252):
    """Walk-forward OOS backtest of a sentiment-tilted fund (equity only).

    Mirrors ``portfolios.oos_backtest`` (same monthly rebalance, same expanding
    estimation window strictly before ``t``, same forward application), but tilts
    the base weights with ``apply_sentiment`` using sentiment dated strictly before
    ``t``. Reuses ``portfolios``' rebalance calendar and weight functions.

    Returns
    -------
    daily_returns : Series
    weights_over_time : DataFrame  (rebalance date x ticker)
    first_oos_date : Timestamp
    lag_audit : DataFrame  (rebalance_date, sentiment_date) proving sentiment_date < t
    """
    weight_fn = portfolios.WEIGHT_FUNCS[method]
    R = returns_wide.sort_index()
    reb_dates = portfolios._month_end_rebalances(R.index, warmup)
    sent = ticker_sentiment_daily.sort_index()

    seg_returns: list[pd.Series] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    lag_rows: list[dict] = []

    for i, t in enumerate(reb_dates):
        past = R.loc[R.index < t]                          # expanding window, strictly before t
        base = weight_fn(past)

        # Lagged sentiment: the last trading day STRICTLY before the rebalance.
        sent_before = sent.loc[sent.index < t]
        sent_date = sent_before.index[-1]
        lagged = sent_before.iloc[-1]                      # Series over tickers
        lag_rows.append({"rebalance_date": t, "sentiment_date": sent_date})

        w = apply_sentiment(base, lagged, lam=lam)
        weight_rows[t] = w

        if i + 1 < len(reb_dates):
            seg = R.loc[(R.index > t) & (R.index <= reb_dates[i + 1])]
        else:
            seg = R.loc[R.index > t]
        if seg.empty:
            continue
        held = seg.reindex(columns=w.index).fillna(0.0)
        seg_returns.append(held.dot(w.values))

    daily_returns = pd.concat(seg_returns).sort_index()
    daily_returns.name = f"{method}+sentiment"
    weights_over_time = (pd.DataFrame(weight_rows).T
                         .reindex(columns=returns_wide.columns)
                         .sort_index())
    weights_over_time.index.name = "rebalance_date"
    lag_audit = pd.DataFrame(lag_rows)
    return daily_returns, weights_over_time, daily_returns.index[0], lag_audit
