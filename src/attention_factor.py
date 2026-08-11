"""Station 3 innovation - "the Noise Tax attention factor" (30% extension).

Motivation. Part A's Noise Tax found that the raw "loud = riskier" claim fails
market-wide: pooled across all 50 stocks, attention concentration does not line up
with realised risk. A weak signal survives only WITHIN sectors - once names are
compared like-for-like, the burstier, more attention-concentrated stocks tilt
slightly riskier. The Old Money thesis ("patience beats hype") turns that into a
portfolio rule: within each sector, tilt AWAY from loud, concentrated-attention
names toward the quiet compounders, holding the sector mix fixed.

This module TRADES that signal, so it must be look-ahead safe: the concentration
used at rebalance t is dated strictly before t (>= 1 trading-day lag), and the
rolling signal itself (``noise_tax.rolling_attention_concentration``) uses only
past headline counts. Equity-only - crypto has no news. The tilt is sector-neutral:
each sector's total weight is unchanged, so the bet is purely within-sector name
selection, not a sector rotation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolios

ZERO_TOL = portfolios.ZERO_TOL


def apply_attention(base_weights: pd.Series, concentration: pd.Series,
                    ticker_sector, lam: float = 0.5) -> pd.Series:
    """Sector-neutral tilt toward LOW attention-concentration (quiet) names.

    Parameters
    ----------
    base_weights : Series
        Per-ticker weights from a ``portfolios`` weight function (sums to 1).
    concentration : Series
        Each ticker's LAGGED concentration score (as of the last trading day before
        the rebalance). Higher = louder / burstier.
    ticker_sector : Series or dict
        Ticker -> sector map.
    lam : float
        Tilt strength (default 0.5).

    Rule
    ----
    Within each sector, z-score the lagged concentration across the sector's HELD
    names, then set ``w_tilted proportional to base_weight * (1 - lam * z)`` (the
    minus overweights LOW concentration), clip at 0 (long-only), and renormalise
    WITHIN the sector so that sector's total weight is unchanged (sector-neutral).
    A sector with fewer than two held names carrying concentration, or no
    cross-sectional spread, is left untouched. Names the optimiser did not hold stay
    out.
    """
    sector_of = pd.Series(ticker_sector)
    out = base_weights.astype(float).copy()
    held = base_weights[base_weights > ZERO_TOL]
    if held.empty:
        return out

    held_sector = sector_of.reindex(held.index)
    for sector, names in held.groupby(held_sector).groups.items():
        bw = held.loc[names]
        sector_total = bw.sum()
        c = concentration.reindex(names)
        valid = c.dropna()
        sd = valid.std()
        if len(valid) < 2 or not np.isfinite(sd) or sd == 0:
            continue  # no usable spread -> leave this sector at its base weights

        z = ((c - valid.mean()) / sd).fillna(0.0)          # missing name -> neutral
        tilted = (bw * (1.0 - lam * z)).clip(lower=0.0)     # overweight LOW concentration
        if tilted.sum() <= 0:
            continue
        tilted = tilted / tilted.sum() * sector_total       # renormalise WITHIN sector
        out.loc[names] = tilted
    return out


def attention_backtest(equity_wide: pd.DataFrame, method: str,
                       concentration_daily: pd.DataFrame, ticker_sector,
                       periods_per_year: int = 252, lam: float = 0.5,
                       warmup: int = 252):
    """Walk-forward OOS backtest of the sector-neutral attention tilt (equity only).

    Mirrors ``fusion.fusion_backtest``: base weights from returns strictly before
    ``t``, concentration as of the last trading day strictly before ``t``, tilt with
    ``apply_attention``, apply forward to the next rebalance. Reuses ``portfolios``'
    rebalance calendar and weight functions.

    Returns daily_returns (Series), weights_over_time (DataFrame), first_oos_date,
    and lag_audit (rebalance_date, signal_date) proving signal_date < t.
    """
    weight_fn = portfolios.WEIGHT_FUNCS[method]
    R = equity_wide.sort_index()
    reb_dates = portfolios._month_end_rebalances(R.index, warmup)
    conc = concentration_daily.sort_index()

    seg_returns: list[pd.Series] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    lag_rows: list[dict] = []

    for i, t in enumerate(reb_dates):
        past = R.loc[R.index < t]
        base = weight_fn(past)

        conc_before = conc.loc[conc.index < t]        # strictly before t -> look-ahead safe
        signal_date = conc_before.index[-1]
        lagged = conc_before.iloc[-1]                  # Series over tickers
        lag_rows.append({"rebalance_date": t, "signal_date": signal_date})

        w = apply_attention(base, lagged, ticker_sector, lam=lam)
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
    daily_returns.name = f"{method}+attention"
    weights_over_time = (pd.DataFrame(weight_rows).T
                         .reindex(columns=equity_wide.columns)
                         .sort_index())
    weights_over_time.index.name = "rebalance_date"
    lag_audit = pd.DataFrame(lag_rows)
    return daily_returns, weights_over_time, daily_returns.index[0], lag_audit
