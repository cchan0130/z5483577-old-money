"""Part A innovation - "The Noise Tax" (DESCRIPTIVE ONLY).

Idea: measure how "loud" each of the 50 equities is from its news ATTENTION
(headline COUNTS, never sentiment scores), then describe whether louder stocks
carry more risk. Equity-only - crypto has no news.

Strictly descriptive: this module counts headlines and correlates the counts with
realised risk. It does NOT score sentiment, form portfolios, build trading
signals, or backtest. Every "z-score" here is a descriptive standardisation, not a
signal to trade.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # equity annualisation factor
SPIKE_Z = 2.0       # a "spike" day is abnormal attention z > 2


def daily_attention(panel: pd.DataFrame, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (ticker, trading_day) headline COUNT on the full trading calendar.

    Attention = number of headlines. Every ticker is reindexed onto the complete
    equity trading calendar so days with no news are explicit zeros (not missing).
    This is attention, not sentiment.

    Returns long: trading_day, ticker, n.
    """
    cal = pd.DatetimeIndex(trading_calendar).sort_values()
    tickers = sorted(panel["ticker"].unique())

    counts = panel.groupby(["ticker", "trading_day"]).size().rename("n")
    idx = pd.MultiIndex.from_product([tickers, cal], names=["ticker", "trading_day"])
    out = counts.reindex(idx, fill_value=0).reset_index()
    return out[["trading_day", "ticker", "n"]]


def _burst_share_window(w: np.ndarray) -> float:
    """Burst share of one trailing window of daily counts (past counts only).

    Threshold = window mean + 2*std (sample). A day is abnormal if its count
    exceeds the threshold. Burst share = counts on abnormal days / total counts in
    the window. Higher = the ticker's attention arrives in concentrated bursts.
    """
    total = w.sum()
    if total <= 0:
        return 0.0
    sd = w.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    thr = w.mean() + SPIKE_Z * sd
    return float(w[w > thr].sum() / total)


def rolling_attention_concentration(attention_counts: pd.DataFrame,
                                    window: int = 252) -> pd.DataFrame:
    """Walk-forward attention-concentration signal (TRADABLE; past counts only).

    ``attention_counts`` is wide: index = trading_day, columns = ticker, values =
    daily headline COUNT (from ``daily_attention`` pivoted wide; no-news days are
    explicit zeros). For each ticker and each date, over the trailing ``window`` of
    daily counts ENDING AT that date, the concentration score is the burst share of
    the window: the share of the ticker's headlines that landed on its own
    abnormal-attention days, where a day is abnormal if its count exceeds the
    window's mean + 2*std (see ``_burst_share_window``).

    Because the rolling window ends at each row and uses ``min_periods=window``, the
    value on date d is a function of counts on days <= d only - never future counts.
    The one-trading-day lag that makes it look-ahead safe for trading is applied by
    the caller (``attention_backtest`` reads the value dated strictly before t).

    Higher = louder / more concentrated attention (a burstier news profile). The
    Old Money tilt overweights the LOW-concentration (quiet) names within a sector.

    Returns wide: index = trading_day, columns = ticker, values = concentration in
    [0, 1] (NaN for the leading ``window - 1`` rows before a full window exists).
    """
    counts = attention_counts.sort_index()
    conc = counts.apply(
        lambda col: col.rolling(window, min_periods=window)
                       .apply(_burst_share_window, raw=True))
    conc.index.name = counts.index.name or "trading_day"
    return conc


def abnormal_attention(attention: pd.DataFrame):
    """Within-ticker standardised attention z = (n - mean_i) / std_i.

    The mean and std are full-sample per ticker (descriptive, not a trading
    signal). A ticker with zero variance gets z = 0.

    Returns:
        (attention_z, summary)
        attention_z: attention + columns z, is_spike (z > 2).
        summary: per ticker spike_rate (share of trading days with z > 2) and
            burst_share (share of the ticker's TOTAL headlines landing on its
            z > 2 days).
    """
    df = attention.copy()
    grp = df.groupby("ticker")["n"]
    mean = grp.transform("mean")
    std = grp.transform("std")  # sample std (ddof=1)
    df["z"] = ((df["n"] - mean) / std.replace(0, np.nan)).fillna(0.0)
    df["is_spike"] = df["z"] > SPIKE_Z

    rows = []
    for ticker, g in df.groupby("ticker"):
        total = g["n"].sum()
        spike_headlines = g.loc[g["is_spike"], "n"].sum()
        rows.append({
            "ticker": ticker,
            "spike_rate": float(g["is_spike"].mean()),
            "burst_share": float(spike_headlines / total) if total > 0 else 0.0,
        })
    return df, pd.DataFrame(rows)


def burstiness(attention: pd.DataFrame) -> pd.DataFrame:
    """Goh-Barabasi burstiness of each ticker's news arrivals, plus a CV cross-check.

    Take the trading days with >= 1 headline, measure the gaps (inter-arrival
    times, in TRADING days) between consecutive news days, then

        B = (std_gap - mean_gap) / (std_gap + mean_gap)

    B ranges from -1 (perfectly regular) through 0 (Poisson-random) to +1 (very
    bursty). Needs >= 2 gaps (>= 3 news days) for a stable std, else B is NaN. The
    coefficient of variation of the daily counts (std/mean over all trading days)
    is returned as a simpler cross-check.

    Returns per ticker: burstiness, cv, n_news_days, mean_gap.
    """
    cal = np.sort(attention["trading_day"].unique())
    pos = pd.Series(np.arange(len(cal)), index=cal)  # trading-day index positions

    rows = []
    for ticker, g in attention.groupby("ticker"):
        g = g.sort_values("trading_day")
        news_days = g.loc[g["n"] >= 1, "trading_day"]
        positions = pos.loc[news_days].to_numpy()

        gaps = np.diff(positions)
        if len(gaps) >= 2:
            mean_gap = gaps.mean()
            std_gap = gaps.std(ddof=0)  # population std, canonical Goh-Barabasi
            denom = std_gap + mean_gap
            b = (std_gap - mean_gap) / denom if denom > 0 else np.nan
        else:
            mean_gap = float(gaps.mean()) if len(gaps) else np.nan
            b = np.nan

        counts = g["n"].to_numpy()
        cv = counts.std(ddof=1) / counts.mean() if counts.mean() > 0 else np.nan
        rows.append({
            "ticker": ticker,
            "burstiness": b,
            "cv": cv,
            "n_news_days": int((g["n"] >= 1).sum()),
            "mean_gap": mean_gap,
        })
    return pd.DataFrame(rows)


def loudness_score(features: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight composite loudness across the 50 tickers.

    Standardise burstiness and spike_rate ACROSS tickers (cross-sectional z-score)
    and average them equal-weight:

        loudness_z = mean(z(burstiness), z(spike_rate))

    An equal-weight composite, deliberately not optimised. Adds burstiness_z,
    spike_rate_z, loudness_z to the input.
    """
    df = features.copy()
    for col in ["burstiness", "spike_rate"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    df["loudness_z"] = df[["burstiness_z", "spike_rate_z"]].mean(axis=1)
    return df


def risk_features(returns_long: pd.DataFrame) -> pd.DataFrame:
    """Per equity ticker: annualised vol, max drawdown, downside deviation (MAR=0).

    - ann_vol = daily std * sqrt(252).
    - max_drawdown = min of (cumulative/cumulative-peak - 1) on (1+r).cumprod();
      a negative number (e.g. -0.45 = a 45% peak-to-trough fall).
    - downside_dev = sqrt(mean(min(r, 0)^2)) * sqrt(252), MAR = 0, denominator over
      ALL periods (target downside deviation), annualised for comparison with
      ann_vol.

    Equity only (crypto has no news, so it is out of scope here).
    """
    eq = returns_long[returns_long["asset_class"] == "equity"]
    rows = []
    for ticker, g in eq.groupby("ticker"):
        r = g.sort_values("date")["ret"]
        cum = (1 + r).cumprod()
        drawdown = cum / cum.cummax() - 1
        downside = np.minimum(r, 0.0)
        rows.append({
            "ticker": ticker,
            "ann_vol": r.std() * np.sqrt(TRADING_DAYS),
            "max_drawdown": drawdown.min(),
            "downside_dev": np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS),
        })
    return pd.DataFrame(rows)


def noise_tax_table(loud_df: pd.DataFrame, risk_df: pd.DataFrame,
                    attention: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """One per-ticker Noise Tax table: attention + loudness + risk.

    Args:
        loud_df: output of ``loudness_score`` (has burstiness, cv, spike_rate,
            burst_share, loudness_z).
        risk_df: output of ``risk_features``.
        attention: output of ``daily_attention`` (for headline volume + coverage).
        sector_map: columns ticker, sector.

    Columns: ticker, sector, avg_daily_headlines, coverage_days_share, burstiness,
    cv, spike_rate, burst_share, loudness_z, ann_vol, max_drawdown, downside_dev.
    """
    vol = attention.groupby("ticker").agg(
        avg_daily_headlines=("n", "mean"),
        coverage_days_share=("n", lambda s: float((s >= 1).mean())),
    ).reset_index()

    keep = ["ticker", "burstiness", "cv", "spike_rate", "burst_share", "loudness_z"]
    table = (loud_df[keep]
             .merge(vol, on="ticker", how="left")
             .merge(risk_df, on="ticker", how="left")
             .merge(sector_map, on="ticker", how="left"))

    cols = ["ticker", "sector", "avg_daily_headlines", "coverage_days_share",
            "burstiness", "cv", "spike_rate", "burst_share", "loudness_z",
            "ann_vol", "max_drawdown", "downside_dev"]
    return table[cols].sort_values("loudness_z", ascending=False).reset_index(drop=True)


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score across the 50 tickers."""
    return (s - s.mean()) / s.std()


def attention_axes(features: pd.DataFrame) -> pd.DataFrame:
    """Split loudness into three comparable per-ticker axes (each a cross-sectional z).

    The original composite conflated two things; separate them so we can re-test:
        - loud_original = the existing composite, kept AS-IS for comparison.
        - loud_volume   = z of avg_daily_headlines - intuitive "how much attention".
        - loud_spike    = z of burst_share - a within-ticker, coverage-normalised
          "how concentrated" (share of a ticker's own headlines on its z>2 days).

    Args:
        features: the noise_tax table (needs ticker, sector, avg_daily_headlines,
            burst_share, loudness_z).

    Returns per ticker: ticker, sector, loud_original, loud_volume, loud_spike.
    """
    return pd.DataFrame({
        "ticker": features["ticker"].values,
        "sector": features["sector"].values,
        "loud_original": features["loudness_z"].values,  # kept as-is
        "loud_volume": _zscore(features["avg_daily_headlines"]).values,
        "loud_spike": _zscore(features["burst_share"]).values,
    })


def confound_check(table: pd.DataFrame) -> pd.DataFrame:
    """How coverage-confounded is each loudness axis?

    Spearman of each variant against coverage_days_share and avg_daily_headlines.
    A value near +/-1 means the axis is largely a coverage proxy; near 0 means it
    is not. Expected columns in ``table``: loud_original, loud_volume, loud_spike,
    coverage_days_share, avg_daily_headlines.

    Returns tidy: variant, confounder, spearman.
    """
    variants = ["loud_original", "loud_volume", "loud_spike"]
    confounders = ["coverage_days_share", "avg_daily_headlines"]
    rows = []
    for v in variants:
        for c in confounders:
            rows.append({
                "variant": v,
                "confounder": c,
                "spearman": table[v].corr(table[c], method="spearman"),
            })
    return pd.DataFrame(rows)


def within_sector_corr(table: pd.DataFrame, x: str, y: str) -> dict:
    """Correlation of x and y after removing sector means (sector-demeaned).

    Subtracts each variable's sector mean, then correlates the residuals. This
    strips out any between-sector effect so what remains is the within-sector
    (like-for-like) association. Returns {'pearson': ..., 'spearman': ...}.
    """
    d = table[["sector", x, y]].dropna().copy()
    xd = d[x] - d.groupby("sector")[x].transform("mean")
    yd = d[y] - d.groupby("sector")[y].transform("mean")
    return {
        "pearson": xd.corr(yd),
        "spearman": xd.corr(yd, method="spearman"),
    }


def _residualise(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Residuals of y after an OLS fit on control (with intercept)."""
    design = np.column_stack([np.ones(len(control)), control])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def partial_corr(table: pd.DataFrame, x: str, y: str, control: str) -> dict:
    """Linear partial correlation of x and y controlling for one variable.

    Residualise x and y on ``control`` (OLS with intercept) and correlate the
    residuals. The Spearman version residualises the RANKS of x and y on the rank
    of the control, so it is robust to monotone non-linearity. Answers "does the
    loudness-risk link survive once coverage is held fixed?".

    Returns {'pearson': ..., 'spearman': ...}.
    """
    d = table[[x, y, control]].dropna()
    rx = _residualise(d[x].to_numpy(), d[control].to_numpy())
    ry = _residualise(d[y].to_numpy(), d[control].to_numpy())
    pearson = float(np.corrcoef(rx, ry)[0, 1])

    dr = d.rank()
    rxr = _residualise(dr[x].to_numpy(), dr[control].to_numpy())
    ryr = _residualise(dr[y].to_numpy(), dr[control].to_numpy())
    spearman = float(np.corrcoef(rxr, ryr)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def loudness_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Headline comparison: each loudness axis vs each risk metric, four ways.

    Rows = loudness variant x risk metric (ann_vol, max_drawdown). Columns:
    pearson_pooled, spearman_pooled, spearman_within_sector, and
    partial_spearman_ctrl_coverage (coverage_days_share held fixed). Reading across
    a row shows whether an axis's link to risk survives de-confounding.

    ``table`` needs the three loud_* axes, ann_vol, max_drawdown, sector, and
    coverage_days_share.
    """
    variants = ["loud_original", "loud_volume", "loud_spike"]
    risks = ["ann_vol", "max_drawdown"]
    rows = []
    for v in variants:
        for rk in risks:
            pair = table[[v, rk]].dropna()
            rows.append({
                "variant": v,
                "risk_metric": rk,
                "pearson_pooled": pair[v].corr(pair[rk]),
                "spearman_pooled": pair[v].corr(pair[rk], method="spearman"),
                "spearman_within_sector": within_sector_corr(table, v, rk)["spearman"],
                "partial_spearman_ctrl_coverage":
                    partial_corr(table, v, rk, "coverage_days_share")["spearman"],
            })
    return pd.DataFrame(rows)


def loudness_risk_correlation(table: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional correlation of loudness measures vs risk, across the 50 stocks.

    Pearson and Spearman of loudness_z and its components (burstiness, spike_rate)
    against ann_vol and max_drawdown. Spearman is the headline (rank-based, robust
    to the fat right tail in attention). Descriptive association only - no causal
    or predictive claim.

    Returns tidy: measure, risk_metric, pearson, spearman, n.
    """
    measures = ["loudness_z", "burstiness", "spike_rate"]
    risks = ["ann_vol", "max_drawdown"]
    rows = []
    for m in measures:
        for rk in risks:
            pair = table[[m, rk]].dropna()
            rows.append({
                "measure": m,
                "risk_metric": rk,
                "pearson": pair[m].corr(pair[rk], method="pearson"),
                "spearman": pair[m].corr(pair[rk], method="spearman"),
                "n": len(pair),
            })
    return pd.DataFrame(rows)
