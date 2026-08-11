"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Long-only funds (weights >= 0, sum to 1) built walk-forward with NO look-ahead:
every rebalance estimates weights from returns STRICTLY BEFORE the rebalance date
and applies them to the days AFTER it. Annualise equities with 252, crypto with
365 (passed in as ``periods_per_year``). See CLAUDE.md and the Part B brief.

Design notes
------------
* A "past-window return matrix" R is a wide frame (date x ticker) of daily simple
  returns. Before optimising we drop any column with a NaN in the window, so only
  assets with a FULL history over the estimation window are investable. This keeps
  the covariance matrix well defined and avoids silently mixing assets that only
  started trading part-way through the window.
* Every weight function returns a pandas Series indexed by the surviving tickers.
  The four functions share the same interface so ``oos_backtest`` can swap them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Weights below this are treated as "not held" (numerical dust from the optimiser).
ZERO_TOL = 1e-6


# --------------------------------------------------------------------------- #
# Universe helper
# --------------------------------------------------------------------------- #
def _investable(R: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with any NaN in the window -> assets with full history only.

    Also drops any column with zero variance in the window (a flat series breaks
    the variance-based optimisers with a singular covariance direction).
    """
    R = R.dropna(axis=1, how="any")
    if R.shape[1] > 1:
        nonflat = R.columns[R.std(axis=0) > 0]
        R = R[nonflat]
    return R


def _as_weights(w: np.ndarray, cols: pd.Index) -> pd.Series:
    """Clip tiny negatives from the optimiser, renormalise, return a tidy Series."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    if total <= 0:  # optimiser collapsed; fall back to equal weight
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    return pd.Series(w, index=cols)


# --------------------------------------------------------------------------- #
# Weight functions  (each takes a past-window R, returns weights >= 0, sum 1)
# --------------------------------------------------------------------------- #
def equal_weight(R: pd.DataFrame) -> pd.Series:
    """1/n over every asset with full history in the window."""
    cols = _investable(R).columns
    n = len(cols)
    return pd.Series(np.repeat(1.0 / n, n), index=cols)


def min_variance(R: pd.DataFrame) -> pd.Series:
    """Global minimum-variance long-only portfolio: min w'Cov w, sum w = 1, w >= 0.

    Solved with SLSQP. x0 is equal weight; the sum-to-1 equality and the [0, 1]
    box give a long-only fully invested solution.
    """
    Ri = _investable(R)
    cols = Ri.columns
    n = len(cols)
    if n == 1:
        return pd.Series([1.0], index=cols)
    cov = Ri.cov().values

    def objective(w):
        return float(w @ cov @ w)

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if res.success else x0
    return _as_weights(w, cols)


def risk_parity(R: pd.DataFrame) -> pd.Series:
    """Long-only risk-parity weights: equalise each asset's risk contribution.

    Asset i's risk contribution is ``w_i * (Cov w)_i``; at the equal-risk-
    contribution point every asset contributes the same share of portfolio
    variance. We minimise the squared spread of contributions about their mean,
    subject to sum w = 1 and a small positive floor (risk parity needs w > 0).
    """
    Ri = _investable(R)
    cols = Ri.columns
    n = len(cols)
    if n == 1:
        return pd.Series([1.0], index=cols)
    cov = Ri.cov().values

    def objective(w):
        port_var = w @ cov @ w
        # Risk-contribution FRACTIONS (they sum to 1); at ERC each equals 1/n.
        # Normalising by port_var keeps the objective O(1) so SLSQP does not
        # declare convergence at x0 on a microscopic raw-variance objective.
        frac = w * (cov @ w) / port_var
        return float(np.sum((frac - 1.0 / n) ** 2))

    x0 = np.repeat(1.0 / n, n)
    bounds = [(1e-6, 1.0)] * n      # strictly positive: a zero weight has no risk contribution
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = res.x if res.success else x0
    return _as_weights(w, cols)


def max_sharpe(R: pd.DataFrame) -> pd.Series:
    """Tangency (max-Sharpe) long-only portfolio with rf = 0.

    Maximise ``(w'mean) / sqrt(w'Cov w)`` by minimising its negative, subject to
    sum w = 1 and w >= 0. mean and Cov are the sample daily moments over the
    window; the ratio is scale-free in the annualisation factor, so rf = 0 here.
    """
    Ri = _investable(R)
    cols = Ri.columns
    n = len(cols)
    if n == 1:
        return pd.Series([1.0], index=cols)
    mu = Ri.mean().values
    cov = Ri.cov().values

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol <= 0:
            return 0.0
        return float(-ret / vol)

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = res.x if res.success else x0
    return _as_weights(w, cols)


WEIGHT_FUNCS = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_sharpe": max_sharpe,
}


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def _month_end_rebalances(index: pd.DatetimeIndex, warmup: int) -> list[pd.Timestamp]:
    """Last actual trading day of each month, keeping only dates AFTER the warm-up.

    ``index`` is the sorted return calendar. We take the last trading day in each
    calendar month, then keep a date only if at least ``warmup`` observations sit
    strictly before it (its 0-based position >= warmup). That first surviving date
    is the earliest point at which a walk-forward estimate has a full warm-up.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    pos = {d: i for i, d in enumerate(idx)}
    last_per_month = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    return [d for d in last_per_month if pos[d] >= warmup]


# --------------------------------------------------------------------------- #
# Walk-forward out-of-sample backtest
# --------------------------------------------------------------------------- #
def oos_backtest(returns_wide: pd.DataFrame, method: str = "min_variance",
                 periods_per_year: int = 252, warmup: int = 252,
                 rebalance: str = "ME"):
    """Walk-forward OOS backtest with a monthly rebalance and no look-ahead.

    Parameters
    ----------
    returns_wide : DataFrame
        Daily simple returns, date index x ticker columns.
    method : str
        One of ``WEIGHT_FUNCS``.
    periods_per_year : int
        252 for equity-calendar funds, 365 for crypto.
    warmup : int
        Minimum observations before the first rebalance (default 252 days).
    rebalance : str
        Kept for signature compatibility; monthly ("ME") is implemented.

    Procedure (guarantees no look-ahead)
    ------------------------------------
    Rebalance dates are the last trading day of each month, starting only after
    the warm-up. At rebalance date ``t`` weights are estimated from the expanding
    window of returns STRICTLY BEFORE ``t`` (``index < t``) and then applied to
    the daily returns AFTER ``t`` up to and including the next rebalance date. A
    weight decided at ``t`` therefore only ever multiplies returns dated later
    than the data that produced it.

    Returns
    -------
    daily_returns : Series
        Daily OOS portfolio returns (index = trading date).
    weights_over_time : DataFrame
        Index = rebalance date, columns = every ticker in ``returns_wide``,
        values = weight (NaN where the asset was outside that window's universe).
    first_oos_date : Timestamp
        First live OOS return date (the first day after the first rebalance).
    """
    if method not in WEIGHT_FUNCS:
        raise ValueError(f"unknown method {method!r}; choose from {list(WEIGHT_FUNCS)}")
    weight_fn = WEIGHT_FUNCS[method]

    R = returns_wide.sort_index()
    reb_dates = _month_end_rebalances(R.index, warmup)
    if not reb_dates:
        raise ValueError("no rebalance dates survive the warm-up; check the calendar length")

    seg_returns: list[pd.Series] = []
    weight_rows: dict[pd.Timestamp, pd.Series] = {}

    for i, t in enumerate(reb_dates):
        past = R.loc[R.index < t]                       # expanding window, strictly before t
        w = weight_fn(past)
        weight_rows[t] = w

        # Apply to days AFTER t, up to (and including) the next rebalance date.
        if i + 1 < len(reb_dates):
            nxt = reb_dates[i + 1]
            seg = R.loc[(R.index > t) & (R.index <= nxt)]
        else:
            seg = R.loc[R.index > t]
        if seg.empty:
            continue
        # A held asset missing on a given day contributes 0 that day (fillna(0)).
        held = seg.reindex(columns=w.index).fillna(0.0)
        seg_returns.append(held.dot(w.values))

    daily_returns = pd.concat(seg_returns).sort_index()
    daily_returns.name = f"{method}"

    weights_over_time = (pd.DataFrame(weight_rows).T
                         .reindex(columns=returns_wide.columns)
                         .sort_index())
    weights_over_time.index.name = "rebalance_date"

    first_oos_date = daily_returns.index[0]
    return daily_returns, weights_over_time, first_oos_date


# --------------------------------------------------------------------------- #
# Fact-sheet metrics
# --------------------------------------------------------------------------- #
def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, annualised vol, Sharpe (rf = 0), and max drawdown.

    * ann_return = mean(daily) * periods_per_year
    * ann_vol    = std(daily) * sqrt(periods_per_year)   (sample std, ddof=1)
    * sharpe     = ann_return / ann_vol                  (rf = 0)
    * max_drawdown = most negative value of (growth / running max - 1),
      where growth is the cumulative product of (1 + daily).
    """
    r = daily_returns.dropna()
    mean, std = r.mean(), r.std()
    ann_return = mean * periods_per_year
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    growth = (1.0 + r).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = drawdown.min()

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }
