"""Reproduce your Part B results. Run from the project root:

    python scripts/run_part_b.py

Station 3: build the daily return matrices, run the walk-forward out-of-sample
backtest for 12 funds (3 families x 4 methods), then build the VADER sector
sentiment index. Writes the fund returns, weights, metrics, and sentiment index
under results/. This is the BUILD step (nltk runs here, never in the app).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import (etl, features, portfolios, quality, sentiment, fusion,  # noqa: E402
                 noise_tax, attention_factor, data_access)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

LAM = 0.5  # sentiment tilt strength (baseline; a naive tilt may not help)
METHODS = ["equal_weight", "min_variance", "risk_parity", "max_sharpe"]
METHOD_LABELS = {
    "equal_weight": "Equal-Weight",
    "min_variance": "Min-Variance",
    "risk_parity": "Risk-Parity",
    "max_sharpe": "Max-Sharpe",
}
FAMILY_LABELS = {"equity": "Equity", "crypto": "Crypto", "combined": "Combined"}


# --------------------------------------------------------------------------- #
# 1) Returns prep
# --------------------------------------------------------------------------- #
def _to_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns per ticker (computed WITHIN the panel), pivoted wide."""
    long = features.daily_returns(prices, price_col="adjClose")
    wide = long.pivot(index="date", columns="ticker", values="ret").sort_index()
    wide.columns.name = None
    return wide


def build_return_matrices():
    """equity_wide (50), crypto_wide (10), combined_wide (equity calendar + crypto).

    Returns are computed within each panel first, THEN merged: combined_wide is
    equity_wide left-joined with crypto_wide onto the EQUITY trading calendar, so
    crypto appears on equity days only (the Part A compute-then-merge rule). No
    return is manufactured across the two calendars.
    """
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()

    equity_wide = _to_wide(eq)
    crypto_wide = _to_wide(cr)
    combined_wide = equity_wide.join(crypto_wide, how="left")  # equity calendar wins

    print(f"[returns] equity_wide {equity_wide.shape}  "
          f"crypto_wide {crypto_wide.shape}  combined_wide {combined_wide.shape}")
    return equity_wide, crypto_wide, combined_wide


# --------------------------------------------------------------------------- #
# 2) 12-fund line-up
# --------------------------------------------------------------------------- #
def run_all_funds():
    equity_wide, crypto_wide, combined_wide = build_return_matrices()
    families = {
        "equity": (equity_wide, 252),
        "crypto": (crypto_wide, 365),
        "combined": (combined_wide, 252),
    }

    fund_returns = {}          # fund name -> daily OOS return Series
    weight_long_rows = []      # tidy long weights
    metric_rows = []           # one row per fund

    for family, (wide, ppy) in families.items():
        for method in METHODS:
            fund = f"{FAMILY_LABELS[family]} {METHOD_LABELS[method]}"
            daily, weights, first_oos = portfolios.oos_backtest(
                wide, method=method, periods_per_year=ppy, warmup=252, rebalance="ME")
            metrics = portfolios.performance_metrics(daily, periods_per_year=ppy)

            fund_returns[fund] = daily

            # Tidy long weights: keep real holdings (weight above numerical dust).
            held_last = weights.iloc[-1].dropna()
            n_assets = int((held_last > portfolios.ZERO_TOL).sum())
            wl = (weights.reset_index()
                         .melt(id_vars="rebalance_date", var_name="ticker", value_name="weight")
                         .dropna(subset=["weight"]))
            wl = wl[wl["weight"] > portfolios.ZERO_TOL].copy()
            wl.insert(1, "fund", fund)
            weight_long_rows.append(wl)

            metric_rows.append({
                "fund": fund,
                "family": family,
                "method": method,
                "ann_return": metrics["ann_return"],
                "ann_vol": metrics["ann_vol"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "n_assets": n_assets,
                "first_oos_date": pd.Timestamp(first_oos).date().isoformat(),
            })

    fund_returns_df = pd.DataFrame(fund_returns).sort_index()
    fund_returns_df.index.name = "date"
    fund_weights_df = pd.concat(weight_long_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    return fund_returns_df, fund_weights_df, metrics_df


def _save(fund_returns_df, fund_weights_df, metrics_df):
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    fund_returns_df.to_csv(RESULTS / "data" / "fund_returns.csv")
    fund_weights_df.to_csv(RESULTS / "data" / "fund_weights.csv", index=False)
    metrics_df.to_csv(RESULTS / "tables" / "performance_metrics.csv", index=False)
    print(f"[save] results/data/fund_returns.csv         {fund_returns_df.shape}")
    print(f"[save] results/data/fund_weights.csv (long)  {fund_weights_df.shape}")
    print(f"[save] results/tables/performance_metrics.csv {metrics_df.shape}")


# --------------------------------------------------------------------------- #
# 3) Sanity checks + reporting
# --------------------------------------------------------------------------- #
def _weights_differ_check(equity_wide):
    """Confirm the four optimisers give DIFFERENT weights on the same window.

    The brief warns optimisers can silently stall and return the x0 (equal
    weight). We estimate all four on one warm-up window and report the max
    absolute gap of each optimiser from equal weight.
    """
    window = equity_wide.iloc[:252]
    ew = portfolios.equal_weight(window)
    print("\n[sanity] max |weight - equal_weight| on the first 252-day equity window:")
    for method in ["min_variance", "risk_parity", "max_sharpe"]:
        w = portfolios.WEIGHT_FUNCS[method](window)
        gap = (w - ew.reindex(w.index)).abs().max()
        print(f"    {method:<13} max gap {gap:.4f}  (n held > tol: {(w > portfolios.ZERO_TOL).sum()})")


# --------------------------------------------------------------------------- #
# 4) VADER sector sentiment index (BUILD-only; nltk runs here, never in the app)
# --------------------------------------------------------------------------- #
def run_sentiment():
    news = etl.load_clean_news()
    calendar = quality.equity_trading_calendar(etl.load_clean_equities())

    scores = sentiment.score_headlines(news, trading_calendar=calendar)
    index = sentiment.sector_sentiment_index(scores, trading_calendar=calendar)

    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    index.to_csv(RESULTS / "data" / "sector_sentiment_index.csv")
    print(f"[save] results/data/sector_sentiment_index.csv  {index.shape}")

    neutral = scores.attrs.get("neutral_share")
    n_scored = scores.attrs.get("n_headlines_scored")
    print(f"\n[sentiment] scored {n_scored:,} headlines; "
          f"VADER-neutral share (compound == 0): {neutral:.1%}")
    print(f"[sentiment] index defined on every trading day: "
          f"{index.notna().all().all()} ({len(index)} days, "
          f"{index.index.min():%Y-%m-%d} to {index.index.max():%Y-%m-%d})")
    print("\n[sentiment] per-sector mean sentiment (full sample):")
    means = index.mean().sort_values(ascending=False)
    for sector, val in means.items():
        print(f"    {sector:<12} {val:+.4f}")
    return index, scores, calendar


# --------------------------------------------------------------------------- #
# 5) Sentiment fusion: tilt the four EQUITY funds (look-ahead safe)
# --------------------------------------------------------------------------- #
def run_fusion(equity_wide, scores, calendar, base_metrics):
    """Tilt each equity fund with lagged sentiment; return appendix funds + compare.

    Reuses the walk-forward structure via fusion.fusion_backtest. Produces one
    "<method> + Sentiment" fund beside each base equity fund and a base-vs-tilt
    comparison table.
    """
    ticker_sent = sentiment.daily_ticker_sentiment(scores, trading_calendar=calendar)

    fusion_returns, weight_long_rows, metric_rows, comp_rows = {}, [], [], []
    lag_audits = []
    for method in METHODS:
        base_fund = f"Equity {METHOD_LABELS[method]}"
        fund = f"{base_fund} + Sentiment"
        daily, weights, first_oos, lag_audit = fusion.fusion_backtest(
            equity_wide, method, ticker_sent, periods_per_year=252, lam=LAM, warmup=252)
        lag_audit["method"] = method
        lag_audits.append(lag_audit)
        m = portfolios.performance_metrics(daily, periods_per_year=252)

        fusion_returns[fund] = daily
        held_last = weights.iloc[-1].dropna()
        wl = (weights.reset_index()
                     .melt(id_vars="rebalance_date", var_name="ticker", value_name="weight")
                     .dropna(subset=["weight"]))
        wl = wl[wl["weight"] > portfolios.ZERO_TOL].copy()
        wl.insert(1, "fund", fund)
        weight_long_rows.append(wl)

        metric_rows.append({
            "fund": fund, "family": "equity", "method": f"{method}+sentiment",
            "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "n_assets": int((held_last > portfolios.ZERO_TOL).sum()),
            "first_oos_date": pd.Timestamp(first_oos).date().isoformat(),
        })

        base = base_metrics.loc[base_metrics["fund"] == base_fund].iloc[0]
        row = {"method": METHOD_LABELS[method]}
        for k in ["sharpe", "ann_return", "ann_vol", "max_drawdown"]:
            row[f"{k}_base"] = float(base[k])
            row[f"{k}_sent"] = m[k]
            row[f"{k}_delta"] = m[k] - float(base[k])
        comp_rows.append(row)

    fusion_returns_df = pd.DataFrame(fusion_returns).sort_index()
    fusion_weights_df = pd.concat(weight_long_rows, ignore_index=True)
    fusion_metrics_df = pd.DataFrame(metric_rows)
    comparison_df = pd.DataFrame(comp_rows)
    lag_audit_all = pd.concat(lag_audits, ignore_index=True)
    return fusion_returns_df, fusion_weights_df, fusion_metrics_df, comparison_df, lag_audit_all


def _lag_check(lag_audit_all, signal_col="sentiment_date", label="sentiment"):
    """Direct look-ahead check: signal used at rebalance t must be dated < t."""
    ok = bool((lag_audit_all[signal_col] < lag_audit_all["rebalance_date"]).all())
    print(f"\n[lag check] every rebalance uses {label} dated strictly before it: {ok}")
    sample = lag_audit_all.drop_duplicates("rebalance_date").tail(3)
    for _, r in sample.iterrows():
        gap = (r["rebalance_date"] - r[signal_col]).days
        print(f"    rebalance {r['rebalance_date']:%Y-%m-%d}  <-  {label} "
              f"{r[signal_col]:%Y-%m-%d}  ({gap} calendar day(s) earlier)")
    return ok


# --------------------------------------------------------------------------- #
# 6) Innovation: the Noise Tax attention factor (sector-neutral, equity-only)
# --------------------------------------------------------------------------- #
def _build_concentration(calendar):
    """Walk-forward attention-concentration signal (wide: trading_day x ticker)."""
    news = etl.load_clean_news()
    panel = features.assemble_headline_panel(news, calendar)
    attention = noise_tax.daily_attention(panel, calendar)
    counts_wide = attention.pivot(index="trading_day", columns="ticker", values="n").fillna(0.0)
    return noise_tax.rolling_attention_concentration(counts_wide, window=252)


def run_attention(equity_wide, calendar, base_metrics, sentiment_metrics):
    """Sector-neutral attention tilt on the four equity funds; compare all three."""
    concentration = _build_concentration(calendar)
    ticker_sector = (data_access.load_sector_universe()
                     .set_index("ticker")["sector"])

    att_returns, weight_long_rows, metric_rows, comp_rows = {}, [], [], []
    lag_audits = []
    for method in METHODS:
        base_fund = f"Equity {METHOD_LABELS[method]}"
        fund = f"{base_fund} + Attention"
        daily, weights, first_oos, lag_audit = attention_factor.attention_backtest(
            equity_wide, method, concentration, ticker_sector,
            periods_per_year=252, lam=LAM, warmup=252)
        lag_audit["method"] = method
        lag_audits.append(lag_audit)
        m = portfolios.performance_metrics(daily, periods_per_year=252)

        att_returns[fund] = daily
        held_last = weights.iloc[-1].dropna()
        wl = (weights.reset_index()
                     .melt(id_vars="rebalance_date", var_name="ticker", value_name="weight")
                     .dropna(subset=["weight"]))
        wl = wl[wl["weight"] > portfolios.ZERO_TOL].copy()
        wl.insert(1, "fund", fund)
        weight_long_rows.append(wl)

        metric_rows.append({
            "fund": fund, "family": "equity", "method": f"{method}+attention",
            "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "n_assets": int((held_last > portfolios.ZERO_TOL).sum()),
            "first_oos_date": pd.Timestamp(first_oos).date().isoformat(),
        })

        base = base_metrics.loc[base_metrics["fund"] == base_fund].iloc[0]
        sent = sentiment_metrics.loc[
            sentiment_metrics["fund"] == f"{base_fund} + Sentiment"].iloc[0]
        row = {"method": METHOD_LABELS[method]}
        for k in ["ann_return", "ann_vol", "sharpe", "max_drawdown"]:
            row[f"{k}_base"] = float(base[k])
            row[f"{k}_sentiment"] = float(sent[k])
            row[f"{k}_attention"] = m[k]
        comp_rows.append(row)

    att_returns_df = pd.DataFrame(att_returns).sort_index()
    att_weights_df = pd.concat(weight_long_rows, ignore_index=True)
    att_metrics_df = pd.DataFrame(metric_rows)
    factor_comparison = pd.DataFrame(comp_rows)
    lag_audit_all = pd.concat(lag_audits, ignore_index=True)

    _sector_neutral_check(equity_wide, concentration, ticker_sector)
    return att_returns_df, att_weights_df, att_metrics_df, factor_comparison, lag_audit_all


def _sector_neutral_check(equity_wide, concentration, ticker_sector):
    """Confirm the tilt is sector-neutral: per-sector total weight is unchanged.

    At the final rebalance, compare base vs tilted sector totals for min_variance
    (a concentrated fund, the hardest case). The max absolute drift should be ~0.
    """
    reb_dates = portfolios._month_end_rebalances(equity_wide.index, 252)
    t = reb_dates[-1]
    base = portfolios.min_variance(equity_wide.loc[equity_wide.index < t])
    lagged = concentration.loc[concentration.index < t].iloc[-1]
    tilted = attention_factor.apply_attention(base, lagged, ticker_sector, lam=LAM)

    sec = pd.Series(ticker_sector).reindex(base.index)
    drift = (tilted.groupby(sec).sum() - base.groupby(sec).sum()).abs().max()
    print(f"\n[sector-neutral check] max |sector total change| at {t:%Y-%m-%d} "
          f"(min_variance): {drift:.2e}  ->  sector-neutral: {drift < 1e-9}")


def main():
    fund_returns_df, fund_weights_df, metrics_df = run_all_funds()

    print("\n[first live OOS dates by family]")
    for family in ["equity", "crypto", "combined"]:
        d = metrics_df.loc[metrics_df["family"] == family, "first_oos_date"].iloc[0]
        print(f"    {family:<9} {d}")

    equity_wide, _, _ = build_return_matrices()
    _weights_differ_check(equity_wide)

    index, scores, calendar = run_sentiment()
    base_metrics = metrics_df.copy()  # the 12 base funds, before any tilt is appended

    # Fusion (required): sentiment tilt on the four equity funds.
    fus_ret, fus_w, fus_m, fusion_comp, fusion_lag = run_fusion(
        equity_wide, scores, calendar, base_metrics)

    # Innovation: the Noise Tax attention factor (sector-neutral).
    att_ret, att_w, att_m, factor_comp, att_lag = run_attention(
        equity_wide, calendar, base_metrics, fus_m)

    # Append both sets of tilted equity funds to the saved outputs (20 funds total).
    fund_returns_df = fund_returns_df.join(fus_ret).join(att_ret)
    fund_weights_df = pd.concat([fund_weights_df, fus_w, att_w], ignore_index=True)
    metrics_df = pd.concat([metrics_df, fus_m, att_m], ignore_index=True)

    _save(fund_returns_df, fund_weights_df, metrics_df)
    fusion_comp.to_csv(RESULTS / "tables" / "fusion_comparison.csv", index=False)
    factor_comp.to_csv(RESULTS / "tables" / "factor_comparison.csv", index=False)
    print(f"[save] results/tables/fusion_comparison.csv   {fusion_comp.shape}")
    print(f"[save] results/tables/factor_comparison.csv   {factor_comp.shape}")

    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print("\n=== factor_comparison (equity: Base vs +Sentiment vs +Attention) ===")
        show = ["method", "sharpe_base", "sharpe_sentiment", "sharpe_attention",
                "ann_return_base", "ann_return_sentiment", "ann_return_attention",
                "ann_vol_base", "ann_vol_sentiment", "ann_vol_attention",
                "max_drawdown_base", "max_drawdown_sentiment", "max_drawdown_attention"]
        print(factor_comp[show].to_string(index=False))

    _lag_check(fusion_lag, signal_col="sentiment_date", label="sentiment")
    _lag_check(att_lag, signal_col="signal_date", label="attention signal")

    # Honest verdict for the attention factor: vs base and vs the sentiment tilt.
    d_base = factor_comp["sharpe_attention"] - factor_comp["sharpe_base"]
    d_sent = factor_comp["sharpe_attention"] - factor_comp["sharpe_sentiment"]
    print(f"\n[attention verdict] lam={LAM}: raised Sharpe vs BASE for "
          f"{int((d_base > 0).sum())}/4 methods (mean delta {d_base.mean():+.4f}); "
          f"vs +SENTIMENT for {int((d_sent > 0).sum())}/4 (mean delta {d_sent.mean():+.4f}).")


if __name__ == "__main__":
    main()
