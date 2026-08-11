# CLAUDE.md - Agent instructions (z5483577, FINS3645 Part B, app "Old Money")

## Project
Part B (Stations 3-4): out-of-sample funds + fact sheets, a VADER sector sentiment index, a sentiment
fusion, my innovation "the Noise Tax attention factor", and a deployed Streamlit app. Reuse my Part A
modules (etl, features, quality, noise_tax, plotstyle). Full brief in PROJECT_BRIEF.md.

## Absolute rules
- NO LOOK-AHEAD. Walk-forward only: portfolio weights and any signal on day t use ONLY data from t-1
  or earlier. Lag sentiment and the attention factor by at least one trading day. This is the top rule.
- Backtest: expanding window with a 252-day warm-up; the out-of-sample period starts after that (state
  the first live date). Monthly rebalance (last trading day). Long-only, weights sum to 1. Risk-free = 0
  (stated). Zero transaction costs baseline (stated); a turnover/cost check is a bonus. Annualise
  equities with 252, crypto with 365.
- Sentiment: VADER via nltk in the BUILD only (one-time nltk.download('vader_lexicon')). Do NOT strip
  casing, punctuation, or negation. Sector index = equal-weight across a sector's tickers; justify how
  no-headline ticker-days are handled. Sentiment work is equity-only.
- The deployed app reads PRECOMPUTED results/ only. streamlit_app.py must NEVER import nltk or recompute
  backtests. Keep it light for the free tier.
- Never commit raw .parquet/.csv (only derived files under results/). Load data only via
  src/data_access.py (frozen).

## Output contract (exact filenames the app and markers rely on)
- results/data/fund_returns.csv, results/data/fund_weights.csv,
  results/data/sector_sentiment_index.csv, results/tables/performance_metrics.csv
- Report figures under results/figures/; other app data under results/data/.

## Build on the stubs
Fill src/portfolios.py, src/sentiment.py, src/fusion.py, scripts/run_part_b.py, and streamlit_app.py.
Reuse Part A modules; do not rewrite them.

## Honesty & verification (from context/verify_ai_output.md)
Never invent a citation, statistic, or source. Flag anything you cannot verify. Show your working for
any number. An honest negative result (e.g. a sentiment tilt that underperforms), explained, is good.

## AI logging convention
At the end of each task, print a draft log entry with only "What I wanted", "Prompt(s)", and "What the
assistant produced" filled. Leave "What was wrong or risky" and "What I changed and why" for me. Do not
create files in ai/.

## Style
Explain non-obvious steps. After writing a function, tell me what could be subtly wrong so I can test it.
