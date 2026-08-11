"""Station 3 fund figures (Part B exhibits).

Reads ONLY the precomputed results/ CSVs (never recomputes the backtest) and
writes four self-contained PNGs under results/figures/. Every figure applies the
"Old Money" plotstyle, labels axes with units, states the sample period, and
carries a caption.

Two calendars: fund_returns.csv unions the equity (252-day) and crypto (365-day)
calendars, so every fund column has structural NaNs (weekends for equity/combined
funds; the pre-2021-02 warm-up gap). We dropna PER FUND COLUMN before any growth
or drawdown maths, so a cumulative product never multiplies across a gap.

    python scripts/make_fund_figures.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from src import plotstyle, data_access  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"

# Method -> a distinct Old Money hue (used wherever methods are compared).
METHOD_COLOR = {
    "Equal-Weight": plotstyle.INK,
    "Min-Variance": plotstyle.HUNTER,
    "Risk-Parity": plotstyle.BRASS,
    "Max-Sharpe": plotstyle.BURGUNDY,
}
FAMILY_COLOR = {
    "equity": plotstyle.INK,
    "crypto": plotstyle.CRYPTO_COLOR,
    "combined": plotstyle.BRASS,
}
METHODS = ["Equal-Weight", "Min-Variance", "Risk-Parity", "Max-Sharpe"]
# Fixed sector order (and colours) so every weights panel stacks consistently.
SECTOR_ORDER = list(plotstyle.SECTOR_COLORS)


def _load():
    r = pd.read_csv(RESULTS / "data" / "fund_returns.csv",
                    parse_dates=["date"], index_col="date")
    w = pd.read_csv(RESULTS / "data" / "fund_weights.csv", parse_dates=["rebalance_date"])
    m = pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")
    return r, w, m


def _growth(series: pd.Series) -> pd.Series:
    """Growth of $1: cumulative product of (1 + daily), on a gap-free series."""
    s = series.dropna()
    return (1.0 + s).cumprod()


def _fmt_period(idx) -> str:
    return f"{idx.min():%d %b %Y} - {idx.max():%d %b %Y}"


# --------------------------------------------------------------------------- #
# 1) Growth of $1, four Combined funds
# --------------------------------------------------------------------------- #
def fig_growth(returns):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    spans = []
    for method in METHODS:
        g = _growth(returns[f"Combined {method}"])
        spans.append(g.index)
        ax.plot(g.index, g.values, color=METHOD_COLOR[method], label=method)
    ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
    ax.set_title("Growth of $1: Combined funds by weighting method")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1 invested (USD, linear scale)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(title="Method", loc="upper left")
    period = _fmt_period(spans[0])
    plotstyle.caption(
        fig,
        f"Figure 1. Out-of-sample growth of a $1 investment (cumulative product of "
        f"1 + daily return) in the four Combined equity+crypto funds, each started "
        f"at its common first live date. Linear y-axis, USD. Monthly rebalanced, "
        f"long-only, no transaction costs. Sample {period}.")
    _save(fig, "growth_of_1_by_method.png")


# --------------------------------------------------------------------------- #
# 2) Drawdown: Combined vs Crypto equal-weight
# --------------------------------------------------------------------------- #
def _drawdown(series: pd.Series) -> pd.Series:
    g = _growth(series)
    return g / g.cummax() - 1.0


def fig_drawdown(returns):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lines = {"Combined Equal-Weight": plotstyle.INK,
             "Crypto Equal-Weight": plotstyle.CRYPTO_COLOR}
    spans = []
    for fund, color in lines.items():
        dd = _drawdown(returns[fund])
        spans.append(dd.index)
        ax.plot(dd.index, dd.values, color=color, label=fund)
        ax.fill_between(dd.index, dd.values, 0.0, color=color, alpha=0.12)
        trough = dd.idxmin()
        ax.annotate(f"{dd.min():.0%}", xy=(trough, dd.min()),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color=color)
    ax.axhline(0.0, color=plotstyle.INK_MUTED, lw=0.8)
    ax.set_title("Drawdown: equity-led vs crypto equal-weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (fraction from running peak)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="lower left")
    period = _fmt_period(min(spans, key=lambda s: s.min()))
    plotstyle.caption(
        fig,
        f"Figure 2. Out-of-sample drawdown (growth of $1 divided by its running "
        f"maximum, minus 1) for the Combined and Crypto equal-weight funds on their "
        f"own trading calendars. The crypto fund troughs near -82% against the "
        f"equity-led Combined fund's -29%. Sample {period}.")
    _save(fig, "drawdown.png")


# --------------------------------------------------------------------------- #
# 3) Equity weights over time, 2x2 stacked areas (top-10 + Other)
# --------------------------------------------------------------------------- #
def _weight_panel(ax, weights_long, ticker_sector, fund, title):
    """Stacked area of sector weight shares over time for one equity fund."""
    df = weights_long[weights_long["fund"] == fund].copy()
    df["sector"] = df["ticker"].map(ticker_sector)
    by_sector = (df.groupby(["rebalance_date", "sector"])["weight"].sum()
                   .unstack("sector")
                   .reindex(columns=SECTOR_ORDER)   # fixed order across panels
                   .fillna(0.0)
                   .sort_index())
    colors = [plotstyle.SECTOR_COLORS[s] for s in SECTOR_ORDER]
    ax.stackplot(by_sector.index, by_sector.T.values, colors=colors,
                 labels=SECTOR_ORDER, edgecolor="none")
    ax.set_ylim(0, 1)
    ax.set_xlim(by_sector.index.min(), by_sector.index.max())
    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", labelrotation=0)


def fig_weights(weights_long):
    ticker_sector = (data_access.load_sector_universe()
                     .set_index("ticker")["sector"].to_dict())  # static lookup, not a recompute
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), METHODS):
        _weight_panel(ax, weights_long, ticker_sector, f"Equity {method}", method)
    for ax in axes[:, 0]:
        ax.set_ylabel("Weight in sector")
    fig.suptitle("Equity portfolio weights over time, by weighting method")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plotstyle.SECTOR_COLORS[s])
               for s in SECTOR_ORDER]
    fig.legend(handles, SECTOR_ORDER, title="Sector", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    period = _fmt_period(
        weights_long.loc[weights_long.fund.str.startswith("Equity"), "rebalance_date"])
    plotstyle.caption(
        fig,
        f"Figure 3. Equity-family portfolio weights at each monthly rebalance, "
        f"aggregated to sector shares as stacked areas (weights sum to 100%; sector "
        f"colours are the shared plotstyle map). Equal-weight and risk-parity stay "
        f"near the fixed sector split; min-variance and max-sharpe concentrate into a "
        f"few defensive sectors. Rebalance dates {period}.")
    _save(fig, "weights_over_time.png", extra_artists=True)


# --------------------------------------------------------------------------- #
# 4) Sharpe barplot across 12 funds, coloured by family
# --------------------------------------------------------------------------- #
def fig_sharpe(metrics):
    fam_order = ["equity", "crypto", "combined"]
    method_key = {"equal_weight": "Equal-Weight", "min_variance": "Min-Variance",
                  "risk_parity": "Risk-Parity", "max_sharpe": "Max-Sharpe"}
    m = metrics[metrics["method"].isin(method_key)].copy()   # 12 base funds only
    m["method_label"] = m["method"].map(method_key)
    piv = m.pivot(index="method_label", columns="family", values="sharpe").reindex(METHODS)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(METHODS))
    width = 0.26
    for i, fam in enumerate(fam_order):
        vals = piv[fam].values
        bars = ax.bar(x + (i - 1) * width, vals, width, color=FAMILY_COLOR[fam],
                      label=fam.capitalize())
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_title("Out-of-sample Sharpe ratio by fund (rf = 0)")
    ax.set_xlabel("Weighting method")
    ax.set_ylabel("Annualised Sharpe ratio (rf = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.legend(title="Family")
    plotstyle.caption(
        fig,
        "Figure 4. Annualised out-of-sample Sharpe ratio (risk-free = 0) for all 12 "
        "funds, grouped by weighting method and coloured by family (equity 252-day, "
        "crypto 365-day, combined 252-day annualisation). Chosen over a risk-return "
        "scatter because it ranks all 12 funds directly. Sample per family first live "
        "date to 31 Dec 2023.")
    _save(fig, "sharpe_barplot.png")


# --------------------------------------------------------------------------- #
# 5) Sector sentiment index over time
# --------------------------------------------------------------------------- #
def fig_sentiment(roll_window=21):
    idx = pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv",
                      parse_dates=["trading_day"], index_col="trading_day")
    smooth = idx.rolling(roll_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for sector in SECTOR_ORDER:
        ax.plot(smooth.index, smooth[sector], color=plotstyle.SECTOR_COLORS[sector],
                label=sector, linewidth=1.3, alpha=0.9)
    ax.axhline(0.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
    ax.set_title("Sector news-sentiment index (VADER compound, equal-weight)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Sentiment ({roll_window}-day rolling mean of daily index)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [plt.Line2D([0], [0], color=plotstyle.SECTOR_COLORS[s], lw=1.6)
               for s in SECTOR_ORDER]
    fig.legend(handles, SECTOR_ORDER, title="Sector", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8)
    period = f"{idx.index.min():%d %b %Y} - {idx.index.max():%d %b %Y}"
    plotstyle.caption(
        fig,
        f"Figure 5. Equity sector news-sentiment index: equal-weight mean of member "
        f"tickers' daily VADER compound scores in [-1, 1], forward-filled between "
        f"headlines and 0 before a ticker's first headline, smoothed with a "
        f"{roll_window}-trading-day rolling mean for readability. Defined on every "
        f"trading day. Sample {period}.")
    _save(fig, "sentiment_index.png", extra_artists=True)


# --------------------------------------------------------------------------- #
# 6) Fusion before/after: equity base vs sentiment-tilted Sharpe
# --------------------------------------------------------------------------- #
def fig_fusion():
    comp = pd.read_csv(RESULTS / "tables" / "fusion_comparison.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(comp))
    width = 0.38
    b1 = ax.bar(x - width / 2, comp["sharpe_base"], width,
                color=plotstyle.INK, label="Base")
    b2 = ax.bar(x + width / 2, comp["sharpe_sent"], width,
                color=plotstyle.BRASS, label="+ Sentiment")
    ax.bar_label(b1, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=7, padding=2)
    ax.set_title("Equity funds: Sharpe before vs after the sentiment tilt")
    ax.set_xlabel("Weighting method")
    ax.set_ylabel("Annualised Sharpe ratio (rf = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["method"])
    ax.legend(title="Fund")
    plotstyle.caption(
        fig,
        "Figure 6. Out-of-sample Sharpe ratio (rf = 0) of each equity fund before "
        "and after a lagged-sentiment tilt (lambda = 0.5, sentiment dated strictly "
        "before each rebalance). The tilt helps the diversified funds (equal-weight, "
        "risk-parity) but hurts the concentrated optimisers (min-variance, "
        "max-sharpe); on average it does not beat the base. Sample from 01 Feb 2021 "
        "to 29 Dec 2023.")
    _save(fig, "fusion_before_after.png")


# --------------------------------------------------------------------------- #
# 7) Factor comparison: Base vs +Sentiment vs +Attention (equity Sharpe)
# --------------------------------------------------------------------------- #
def fig_factor():
    comp = pd.read_csv(RESULTS / "tables" / "factor_comparison.csv")
    variants = [("sharpe_base", "Base", plotstyle.INK),
                ("sharpe_sentiment", "+ Sentiment", plotstyle.BRASS),
                ("sharpe_attention", "+ Attention", plotstyle.HUNTER)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(comp))
    width = 0.26
    for i, (col, label, color) in enumerate(variants):
        bars = ax.bar(x + (i - 1) * width, comp[col], width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_title("Equity funds: Sharpe under Base, Sentiment tilt, and the Attention factor")
    ax.set_xlabel("Weighting method")
    ax.set_ylabel("Annualised Sharpe ratio (rf = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["method"])
    ax.set_ylim(0, comp[["sharpe_base", "sharpe_sentiment", "sharpe_attention"]].values.max() * 1.18)
    ax.legend(title="Fund", loc="upper right", ncol=3)
    plotstyle.caption(
        fig,
        "Figure 7. Out-of-sample Sharpe ratio (rf = 0) of each equity fund under the "
        "base weighting, the required sentiment tilt, and the innovation - the "
        "sector-neutral Noise Tax attention factor (both lambda = 0.5, signals dated "
        "strictly before each rebalance). The sector-neutral attention tilt modestly "
        "beats base for 3 of 4 methods and avoids the large min-variance / max-sharpe "
        "losses of the naive sentiment tilt. Sample 01 Feb 2021 to 29 Dec 2023.")
    _save(fig, "factor_comparison.png")


def _save(fig, name, extra_artists=False):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight")
    if extra_artists:
        kw["bbox_extra_artists"] = fig.legends
    fig.savefig(FIGDIR / name, **kw)
    plt.close(fig)
    print(f"[figure] results/figures/{name}")


def main():
    plotstyle.apply_style()
    returns, weights_long, metrics = _load()
    fig_growth(returns)
    fig_drawdown(returns)
    fig_weights(weights_long)
    fig_sharpe(metrics)
    fig_sentiment()
    fig_fusion()
    fig_factor()


if __name__ == "__main__":
    main()
