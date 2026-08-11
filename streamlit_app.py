"""Old Money - a beginner-first, goal-based multi-asset fund app (Station 4).

The deployed app reads ONLY the precomputed CSVs under results/. It never imports the
sentiment, fusion, or portfolios build modules, never loads the VADER lexicon, and
never recomputes a backtest - all funds, weights, metrics, and the sentiment index
are built offline by scripts/run_part_b.py. Blending precomputed daily fund returns
into a user allocation is a weighted sum (not a backtest), which is fine and stays
light for the free tier. Only src.plotstyle is imported, purely for the Old Money
matplotlib look.

Run locally:   streamlit run streamlit_app.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src import plotstyle  # noqa: E402  (matplotlib-only; no data/model imports)

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"

plotstyle.apply_style()
st.set_page_config(page_title="Old Money", layout="wide")

FAMILY_COLOR = {"equity": plotstyle.INK, "crypto": plotstyle.CRYPTO_COLOR,
                "combined": plotstyle.BRASS}
SECTOR_ORDER = list(plotstyle.SECTOR_COLORS)


# --------------------------------------------------------------------------- #
# Cached loaders - the ONLY inputs are the precomputed results/ CSVs
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=86_400, show_spinner="Loading precomputed results...")
def load_metrics():
    return pd.read_csv(RESULTS / "tables" / "performance_metrics.csv")


@st.cache_data(ttl=86_400)
def load_fund_returns():
    return pd.read_csv(RESULTS / "data" / "fund_returns.csv",
                       parse_dates=["date"], index_col="date")


@st.cache_data(ttl=86_400)
def load_fund_weights():
    return pd.read_csv(RESULTS / "data" / "fund_weights.csv",
                       parse_dates=["rebalance_date"])


@st.cache_data(ttl=86_400)
def load_sentiment_index():
    return pd.read_csv(RESULTS / "data" / "sector_sentiment_index.csv",
                       parse_dates=["trading_day"], index_col="trading_day")


@st.cache_data(ttl=86_400)
def load_factor_comparison():
    return pd.read_csv(RESULTS / "tables" / "factor_comparison.csv")


@st.cache_data(ttl=86_400)
def load_fusion_comparison():
    return pd.read_csv(RESULTS / "tables" / "fusion_comparison.csv")


# --------------------------------------------------------------------------- #
# Small pure helpers (weighted sums and cumulative products - never a backtest)
# --------------------------------------------------------------------------- #
def growth_of_one(daily: pd.Series) -> pd.Series:
    r = daily.dropna()
    return (1.0 + r).cumprod()


def drawdown(daily: pd.Series) -> pd.Series:
    g = growth_of_one(daily)
    return g / g.cummax() - 1.0


def ppy_for_families(families) -> int:
    """365 only if every selected fund is crypto (365-day calendar), else 252."""
    fam = set(families)
    return 365 if fam == {"crypto"} else 252


def summarise(daily: pd.Series, ppy: int) -> dict:
    r = daily.dropna()
    mean, std = r.mean(), r.std()
    ann_return = mean * ppy
    ann_vol = std * np.sqrt(ppy)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": drawdown(r).min()}


def pct(x):
    return "-" if pd.isna(x) else f"{x:.1%}"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
metrics = load_metrics()
returns = load_fund_returns()
weights = load_fund_weights()
fund_family = dict(zip(metrics["fund"], metrics["family"]))
ALL_FUNDS = metrics["fund"].tolist()

st.title("Old Money")
st.markdown(
    "**Patience beats hype.** Old Money is a beginner-first, goal-based home for "
    "simple multi-asset funds. Pick a goal, read a one-page fact sheet, and blend a "
    "few funds into an allocation - no jargon, no day-trading. Every fund is a "
    "transparent, monthly-rebalanced, long-only rule (equity, crypto, or a combined "
    "mix), built and frozen offline; this app only *reads* the results.")
st.markdown(
    ":link: **Links:** live app - https://z5483577-old-money.streamlit.app  |  "
    "code - https://github.com/cchan0130/z5483577-old-money")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 - Compare funds", "2 - Fact sheet", "3 - Build an allocation", "4 - Analytics"])


# --------------------------------------------------------------------------- #
# 1) Compare funds
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Compare the funds")
    families = ["equity", "crypto", "combined"]
    chosen = st.multiselect("Filter by family", families, default=families)
    view = metrics[metrics["family"].isin(chosen)].copy()

    show = view[["fund", "family", "method", "ann_return", "ann_vol", "sharpe",
                 "max_drawdown", "n_assets", "first_oos_date"]].copy()
    st.dataframe(
        show.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}",
                           "sharpe": "{:.2f}", "max_drawdown": "{:.1%}"}),
        width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Risk vs return** (annualised, coloured by family; "
                    "base methods labelled, sentiment/attention tilts shown unlabelled)")
        base_methods = {"equal_weight", "min_variance", "risk_parity", "max_sharpe"}
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for fam in chosen:
            sub = view[view["family"] == fam]
            ax.scatter(sub["ann_vol"], sub["ann_return"], s=42,
                       color=FAMILY_COLOR[fam], label=fam, alpha=0.85, edgecolor="none")
        # Label only the four base methods per family - the +sentiment/+attention
        # variants sit on top of their base in the dense equity cluster.
        for _, row in view.iterrows():
            if row["method"] not in base_methods:
                continue
            ax.annotate(row["method"].replace("_", " ").title(),
                        (row["ann_vol"], row["ann_return"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points",
                        color=plotstyle.INK_MUTED)
        ax.set_xlabel("Annualised volatility")
        ax.set_ylabel("Annualised return")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if chosen:
            ax.legend(title="Family", fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_b:
        fam_overlay = st.selectbox("Growth of $1 - choose a family", families, index=0)
        funds_in = metrics.loc[metrics["family"] == fam_overlay, "fund"].tolist()
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for fund in funds_in:
            g = growth_of_one(returns[fund])
            ax.plot(g.index, g.values, linewidth=1.4, label=fund.split(" ", 1)[-1])
        ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value of $1 (linear)")
        plotstyle.format_date_axis(ax)
        ax.legend(fontsize=7, title=fam_overlay.capitalize())
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 2) Fact sheet
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Fund fact sheet")
    fund = st.selectbox("Pick a fund", ALL_FUNDS, index=0)
    row = metrics.loc[metrics["fund"] == fund].iloc[0]
    daily = returns[fund].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", pct(row["ann_return"]))
    c2.metric("Annualised volatility", pct(row["ann_vol"]))
    c3.metric("Sharpe (rf = 0)", f"{row['sharpe']:.2f}")
    c4.metric("Max drawdown", pct(row["max_drawdown"]))
    st.caption(f"Family: {row['family']}  |  method: {row['method']}  |  "
               f"{int(row['n_assets'])} holdings at the last rebalance  |  "
               f"first live date {row['first_oos_date']}  |  "
               f"sample to {daily.index.max():%d %b %Y}.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Growth of $1**")
        g = growth_of_one(daily)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(g.index, g.values, color=plotstyle.INK, linewidth=1.6)
        ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
        ax.set_ylabel("Value of $1")
        plotstyle.format_date_axis(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown("**Drawdown**")
        dd = drawdown(daily)
        fig, ax = plt.subplots(figsize=(5.4, 2.6))
        ax.plot(dd.index, dd.values, color=plotstyle.BURGUNDY, linewidth=1.3)
        ax.fill_between(dd.index, dd.values, 0.0, color=plotstyle.BURGUNDY, alpha=0.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        plotstyle.format_date_axis(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:
        st.markdown("**Current holdings** (latest rebalance)")
        wf = weights[weights["fund"] == fund]
        if wf.empty:
            st.info("No holdings recorded for this fund.")
        else:
            last_date = wf["rebalance_date"].max()
            top = (wf[wf["rebalance_date"] == last_date]
                   .sort_values("weight", ascending=False).head(10))
            st.caption(f"As of {last_date:%d %b %Y} - top {len(top)} of "
                       f"{(wf['rebalance_date'] == last_date).sum()} positions.")
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            ax.barh(top["ticker"][::-1], top["weight"][::-1], color=plotstyle.HUNTER)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_xlabel("Weight")
            st.pyplot(fig, width="stretch")
            plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) Build an allocation (weighted sum of precomputed returns - NOT a backtest)
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Build your allocation")
    st.caption("This blends the funds' PRECOMPUTED daily returns as a weighted sum. "
               "It is an illustration of mixing existing funds, not a new backtest.")
    default = [f for f in ["Combined Equal-Weight", "Equity Min-Variance",
                           "Crypto Risk-Parity"] if f in ALL_FUNDS]
    picks = st.multiselect("Choose 2-5 funds to blend", ALL_FUNDS, default=default)

    if len(picks) < 2:
        st.info("Pick at least two funds to build an allocation.")
    else:
        raw = {}
        cols = st.columns(len(picks))
        for col, fund in zip(cols, picks):
            raw[fund] = col.slider(fund, 0, 100, 100 // len(picks), key=f"w_{fund}")
        total = sum(raw.values())
        if total == 0:
            st.warning("Set at least one weight above zero.")
        else:
            w = {f: v / total for f, v in raw.items()}
            st.write("**Normalised weights:** " +
                     "  ".join(f"{f} {w[f]:.0%}" for f in picks))

            sub = returns[picks].dropna()          # common dates across the picks
            blended = sub.mul(pd.Series(w)).sum(axis=1)
            ppy = ppy_for_families(fund_family[f] for f in picks)
            s = summarise(blended, ppy)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Annualised return", pct(s["ann_return"]))
            m2.metric("Annualised volatility", pct(s["ann_vol"]))
            m3.metric("Sharpe (rf = 0)", f"{s['sharpe']:.2f}")
            m4.metric("Max drawdown", pct(s["max_drawdown"]))
            st.caption(f"Blended over {len(sub):,} common trading days "
                       f"({sub.index.min():%d %b %Y} to {sub.index.max():%d %b %Y}); "
                       f"annualised with {ppy} periods/year.")

            g = growth_of_one(blended)
            fig, ax = plt.subplots(figsize=(9.0, 3.6))
            ax.plot(g.index, g.values, color=plotstyle.BRASS, linewidth=1.8,
                    label="Your allocation")
            ax.axhline(1.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
            ax.set_ylabel("Value of $1")
            plotstyle.format_date_axis(ax)
            ax.legend(fontsize=8)
            st.pyplot(fig, width="stretch")
            plt.close(fig)


# --------------------------------------------------------------------------- #
# 4) Analytics
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Sentiment and the Noise Tax attention factor")

    st.markdown("**Sector news-sentiment index** (VADER compound, equal-weight; "
                "21-day rolling mean for readability)")
    sent = load_sentiment_index()
    smooth = sent.rolling(21, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for sector in SECTOR_ORDER:
        if sector in smooth.columns:
            ax.plot(smooth.index, smooth[sector], color=plotstyle.SECTOR_COLORS[sector],
                    linewidth=1.2, label=sector, alpha=0.9)
    ax.axhline(0.0, color=plotstyle.INK_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sentiment (rolling mean)")
    plotstyle.format_date_axis(ax)
    fig.legend(SECTOR_ORDER, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
               title="Sector")
    st.pyplot(fig, width="stretch", bbox_inches="tight")
    plt.close(fig)

    st.divider()
    st.markdown("**Base vs + Sentiment vs + Attention** (equity funds, Sharpe, rf = 0)")
    fc = load_factor_comparison()
    variants = [("sharpe_base", "Base", plotstyle.INK),
                ("sharpe_sentiment", "+ Sentiment", plotstyle.BRASS),
                ("sharpe_attention", "+ Attention", plotstyle.HUNTER)]
    x = np.arange(len(fc))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for i, (col, label, color) in enumerate(variants):
        bars = ax.bar(x + (i - 1) * width, fc[col], width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(fc["method"])
    ax.set_ylabel("Sharpe ratio (rf = 0)")
    ax.set_ylim(0, fc[[c for c, _, _ in variants]].values.max() * 1.18)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    d_base = (fc["sharpe_attention"] - fc["sharpe_base"])
    d_sent = (fc["sharpe_attention"] - fc["sharpe_sentiment"])
    st.markdown(
        "**What is the Noise Tax attention factor?** For each stock it measures how "
        "*concentrated* its news attention is - how much of its coverage arrives in "
        "abnormal bursts rather than a steady drip. The Old Money tilt leans, *within "
        "each sector*, toward the quiet compounders and away from the loud, "
        "burst-heavy names, keeping every sector's total weight unchanged. The signal "
        "uses only headline counts dated before each rebalance, so it never looks "
        "ahead.")
    st.markdown(
        f"**Honest finding.** The tilt is a *weak within-sector edge*: it raised the "
        f"Sharpe ratio for {int((d_base > 0).sum())} of 4 equity methods versus base "
        f"(average {d_base.mean():+.3f}). Its real advantage is robustness - by "
        f"staying sector-neutral it avoids the large losses the simpler sentiment "
        f"tilt caused on the concentrated funds, beating that sentiment tilt by "
        f"{d_sent.mean():+.3f} Sharpe on average. A small, honest improvement, not a "
        f"silver bullet.")
