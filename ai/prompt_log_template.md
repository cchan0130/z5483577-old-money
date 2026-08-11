# Prompt log - <task name>

## What I wanted
<the goal in one or two sentences>

## Prompt(s)
<the prompt(s) you gave the assistant>

## What the assistant produced
<a summary, or the code/text it returned>

## What was wrong or risky
<bugs, look-ahead, wrong assumptions, hallucinated APIs - what you found>

## What I changed and why
<your correction, in your own words>

---

## Prompt log 1 - Part B setup (carry-over + CLAUDE.md)

### What I wanted
- Set up Part B: carry my reusable Part A modules into src/, replace the placeholder CLAUDE.md with my
  Part B agent instructions, and confirm check_handin passes.

### Prompt(s)
- [Part B setup. This is Stations 3-4 for my app "Old Money", reusing my Part A foundation. The folder is
  already named z5483577_projectB.

1) Carry over my reusable Part A modules into src/. My Part A folder is a sibling at
   ../z5483577_projectA/. Copy these files from that folder's src/ into this project's src/,
   overwriting the stub features.py:
     etl.py, features.py, quality.py, noise_tax.py, plotstyle.py, textstats.py, screens.py
   Do NOT overwrite src/data_access.py (frozen), and do NOT touch the new Part B stubs
   src/portfolios.py, src/sentiment.py, src/fusion.py. If you cannot reach the sibling folder from your
   sandbox, stop and tell me, and I will copy the seven files manually in PyCharm. After copying,
   confirm `from src import features, quality, noise_tax, plotstyle` imports cleanly.

2) Replace the placeholder CLAUDE.md with the content below, then run `python scripts/check_handin.py`
   and show the output.

--- BEGIN CLAUDE.md ---
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
--- END CLAUDE.md ---

After you finish, remind me to reword the CLAUDE.md in my own voice before hand-in.]

### What the assistant produced
- Copied 7 Part A modules (etl, features, quality, noise_tax, plotstyle, textstats, screens) into src/,
  overwriting the features stub; left data_access and the Part B stubs (portfolios, sentiment, fusion)
  untouched; confirmed imports work; wrote the Part B CLAUDE.md; ran check_handin (16 passed, exit 0,
  WARNs only for outputs not built yet).

### What was wrong or risky
- The carried-over features.py has a different assemble_headline_panel signature, so my Part B code must call my version, not the stub's since a
  mismatch would break run_part_b.py.
- Reused Part A modules assume Part A's cleaning conventions and column names, so they need re-checking
  in the Part B pipeline rather than being assumed correct.
- The assistant reached into the sibling Part A folder to copy files and I confirmed it copied only the
  seven I named and left the frozen data_access and the Part B stubs untouched.

### What I changed and why
- I confirmed all imports are OK and Kept the CLAUDE.md as drafted because I read and agree with every rule.
- I will note in AI_NOTES that it was AI-drafted and refined by me, rather than reword it.

---

# Prompt log 2 - Station 3 (funds + out-of-sample backtest engine)

## What I wanted
- Implement Station 3: build equity/crypto/combined daily-return matrices (compute-
  then-merge onto the equity calendar), implement equal_weight, min_variance,
  risk_parity and max_sharpe weight functions plus a walk-forward, no-look-ahead
  oos_backtest and performance_metrics in src/portfolios.py, then run the 12-fund
  line-up in scripts/run_part_b.py and save fund_returns.csv, fund_weights.csv and
  performance_metrics.csv under results/.


## Prompt(s)
- [Station 3 funds + out-of-sample backtest engine. Reuse my Part A modules (etl, features). Follow
  CLAUDE.md: NO LOOK-AHEAD, walk-forward, weights from past data only.

1) Returns prep (in scripts/run_part_b.py or a small helper):
   - Load clean equities and crypto via etl.load_clean_equities / load_clean_crypto.
   - Compute daily simple returns per panel via features.daily_returns (adjClose).
   - Build three wide return matrices (date x ticker), returns computed WITHIN each panel first:
     equity_wide (50 cols), crypto_wide (10 cols), and combined_wide = equity_wide left-joined with
     crypto_wide onto the EQUITY trading calendar (crypto on equity days only; reuse the Part A
     compute-then-merge rule).

2) In src/portfolios.py implement the engine (document each; long-only, weights sum to 1):
   - Weight functions from a past-window return matrix R (drop columns with any NaN in the window so
     only assets with full history are used):
       * equal_weight(R): 1/n.
       * min_variance(R): minimise w'Cov w s.t. sum w = 1, w >= 0 (scipy SLSQP).
       * risk_parity(R): weights that equalise each asset's risk contribution, long-only, sum 1.
       * max_sharpe(R): tangency, maximise (w'mean)/sqrt(w'Cov w) s.t. sum w = 1, w >= 0 (scipy).
   - oos_backtest(returns_wide, method, periods_per_year, warmup=252, rebalance="ME"):
       * Rebalance dates = last trading day of each month, starting only AFTER the 252-day warm-up.
       * At each rebalance date t, estimate weights from returns STRICTLY BEFORE t (expanding window),
         then apply those weights to the daily returns AFTER t up to the next rebalance. This guarantees
         no look-ahead.
       * Return: daily OOS portfolio returns (Series), weights_over_time (DataFrame indexed by rebalance
         date, columns = tickers), and the first live OOS date.
   - performance_metrics(daily_returns, periods_per_year): annualised return (mean * periods),
     annualised vol (std * sqrt(periods)), Sharpe (rf = 0), max drawdown (from growth of $1).

3) In scripts/run_part_b.py, build the line-up: families {equity (252), crypto (365), combined (252)}
   x methods {equal_weight, min_variance, risk_parity, max_sharpe} = 12 funds. Each (family, method) is
   one fund, e.g. "Equity Min-Variance". Run oos_backtest + performance_metrics for each and save:
   - results/data/fund_returns.csv: index = date, one column per fund (daily OOS returns).
   - results/data/fund_weights.csv: tidy long - rebalance_date, fund, ticker, weight (all rebalances).
   - results/tables/performance_metrics.csv: one row per fund - fund, family, method, ann_return,
     ann_vol, sharpe, max_drawdown, n_assets, first_oos_date.

When done: (a) print the performance_metrics table and the first live backtest date, and confirm the
warm-up/rebalance logic uses only past data; (b) sanity-check that weights actually DIFFER across
methods (the brief warns optimisers can silently stall) and name the function most likely to have a
subtle bug + how to test it; (c) print a draft log entry with only "What I wanted", "Prompt(s)", "What
the assistant produced" filled - leave the last two for me, no files in ai/.]

## What the assistant produced
- src/portfolios.py: four weight functions (equal_weight, min_variance, risk_parity, max_sharpe, all
  long-only, sum to 1), an _investable helper (drops assets without full history in the window),
  _month_end_rebalances, oos_backtest (walk-forward, no look-ahead), and performance_metrics
  (ann return, ann vol, Sharpe with rf=0, max drawdown).
- Rewrote scripts/run_part_b.py: builds the three return matrices (equity_wide 50, crypto_wide 10,
  combined_wide 60 via compute-then-merge onto the equity calendar), runs the 12-fund line-up
  (3 families x 4 methods), and saves results/data/fund_returns.csv, results/data/fund_weights.csv
  (tidy long), and results/tables/performance_metrics.csv.
- First run exposed risk_parity silently returning equal weight (raw objective ~1e-11, SLSQP stalled
  at x0); fixed by normalising to risk-contribution fractions, after which all four methods give
  distinct weights.
- Verified: weights sum to 1 and are long-only; the first equity estimation window ends 2021-01-28,
  strictly before the 2021-01-29 rebalance (270 >= 252 warm-up); first live OOS is 2021-02-01
  (equity/combined) and 2020-10-01 (crypto); max_sharpe in-sample-dominates equal weight
  (0.1079 vs 0.0349).

## What was wrong or risky
- risk_parity silently stalled: SLSQP declared convergence at the equal-weight start because the raw
  risk-contribution objective was ~1e-11, so Equity/Combined Risk-Parity came out identical to
  Equal-Weight. Exactly the solver-stall the brief warns about; only my weights-differ sanity check
  caught it.
- max_sharpe is fragile: it concentrates into ~6 names and, despite dominating equal-weight in-sample
  (0.108 vs 0.035), its OOS Sharpe (0.72) is below equal-weight (0.86) - the classic estimation-error
  problem. I will report this honestly.
- fund_returns.csv has structural NaNs (crypto on the 365-day calendar from 2020-10, equity/combined on
  the 252-day calendar from 2021-02), so the app and figures must dropna per fund column, not treat
  these as missing data.

## What I changed and why
- Fixed risk_parity by normalising the objective to risk-contribution fractions (an O(1) objective);
  all four methods then give distinct weights (gaps from equal weight: min-var 0.21, risk-parity 0.02,
  max-sharpe 0.27).
- Verified no look-ahead directly (est window ends 2021-01-28, strictly before the 2021-01-29
  rebalance; weights index<t applied to (t, next]); confirmed weights sum to 1 and long-only.
- Kept only real holdings (weight > 1e-6) in fund_weights.csv so current holdings reflect actual
  positions.

---

# Prompt log 3 - Station 3 fund figures (Part B exhibits)

## What I wanted
- Four self-contained Part B fund figures under results/figures/, reusing the
  plotstyle design system and reading ONLY the precomputed results CSVs (no
  backtest recompute): (1) growth of $1 for the four Combined funds, (2) drawdown
  of Combined vs Crypto equal-weight, (3) a 2x2 equity weights-over-time small-
  multiple, (4) a Sharpe barplot across all 12 funds coloured by family. Each with
  title, labelled axes/units, sample period, caption, and correct two-calendar
  dropna handling.

## Prompt(s)
- [Station 3 fund figures (required Part B exhibits). Reuse plotstyle (apply_style + palette). Read the
precomputed results/data/fund_returns.csv, results/data/fund_weights.csv, and
results/tables/performance_metrics.csv - do NOT recompute the backtest. Each figure self-contained
(title, labelled axes + units, sample period, caption). Handle the two calendars: dropna per fund
column before plotting.

1) results/figures/growth_of_1_by_method.png: growth of $1 (cumprod of 1+daily) for the four COMBINED
   funds, all from their common first OOS date. One line per method, legend, state linear/log y.

2) results/figures/drawdown.png: drawdown curve (growth/running-max - 1) for Combined Equal-Weight and
   Crypto Equal-Weight together, to contrast the ~-29% equity-led drawdown with crypto's ~-82%.

3) results/figures/weights_over_time.png: portfolio weights over time across the four methods for the
   EQUITY family - a 2x2 small-multiple, each panel a stacked area (or top-10 holdings) of that method's
   weights_over_time, so equal-weight looks flat and min-variance/max-sharpe visibly concentrated.

4) results/figures/sharpe_barplot.png: a Sharpe barplot across all 12 funds, grouped/coloured by family
   (equity, crypto, combined). (Or a risk-return scatter of ann_vol vs ann_return coloured by family if
   it reads better - state which you chose.)

When done: (a) list the figures and confirm each uses the design system and has a caption; (b) name the
riskiest bit (I expect the two-calendar / dropna handling); (c) print a draft log entry with sections
1-3 filled, leave the last two for me.]

## What the assistant produced
- scripts/make_fund_figures.py, which reads ONLY the precomputed results/ CSVs (no backtest recompute),
  applies the plotstyle design system, and writes four self-contained PNGs to results/figures/.
- growth_of_1_by_method.png: growth of $1 for the four Combined funds from their common first OOS date
  (2021-02-01), one line per method, linear y.
- drawdown.png: drawdown of Combined vs Crypto equal-weight, each on its own calendar (crypto from
  2020-10, combined from 2021-02), with the -82% and -29% troughs annotated.
- weights_over_time.png: a 2x2 equity small-multiple of weights over time; first drawn with top-10
  tickers (repeated palette colours were misleading) then revised to sector-grouped stacked areas using
  the shared SECTOR_COLORS map, with the ticker->sector lookup from the frozen data_access loader.
- sharpe_barplot.png: Sharpe across all 12 funds, grouped and coloured by family.
- dropna per fund column before every cumulative-product/drawdown calculation (two-calendar handling);
  all four PNGs verified visually, not just that the code ran.

## What was wrong or risky
- The weights small-multiple first used each panel's top-10 tickers on a repeating 10-colour palette,
  so the same colour meant different tickers across panels - misleading. The assistant caught this and
  switched to sector-grouped stacked areas on the shared SECTOR_COLORS map; I reviewed before/after and
  agree the sector version is clearer and correct.
- fund_returns.csv unions the two calendars (equity 252-day, crypto 365-day), so every column has
  structural NaNs; a cumulative product across those gaps would compound a wrong series.
- A subtle failure remains: an interior NaN (a genuine missing day, not a calendar gap) would be
  silently bridged by dropna - worth an explicit "no interior gaps" assert if the fund line-up changes.

## What I changed and why
- Reviewed all four figures myself rather than trusting that the code ran, and confirmed the weights fix reads correctly 
  and the drawdown keeps each fund on its own calendar. 
- Confirmed every figure reads only the precomputed CSVs and applies the design system.

---

# Prompt log 4 - Station 3 VADER sector sentiment index

## What I wanted
- Build the equity-only VADER sentiment index as a BUILD step (nltk here, never in the app). In
  src/sentiment.py: score_headlines (assemble the daily panel on the equity trading calendar, score each
  RAW headline's VADER compound, aggregate to a per-(ticker, day) mean, report the exactly-0 neutral
  share) and sector_sentiment_index (forward-fill each ticker across the calendar, 0 before its first
  headline, then equal-weight sector means, wide with 10 sector columns). Wire into run_part_b.py, save
  results/data/sector_sentiment_index.csv, print neutral share / per-sector means / date range, and add
  results/figures/sentiment_index.png.

## Prompt(s)
- [Station 3 sentiment index (required). Reuse features.assemble_headline_panel and
quality.equity_trading_calendar. This is a BUILD step (nltk allowed here, never in the app). Follow
CLAUDE.md: do NOT strip casing/punctuation/negation; equity-only; the lag is applied later in the fusion.

1) One-time setup: nltk.download('vader_lexicon') guarded so it only downloads if missing. Ensure nltk
   is in requirements-dev.txt (not requirements.txt).

2) In src/sentiment.py implement:
   - score_headlines(panel): build the daily text panel (assemble_headline_panel with the equity
     trading calendar), score each RAW headline with VADER's compound score, and aggregate to a
     per-(ticker, trading_day) score = mean compound of that ticker's headlines that day. Also report
     the share of headlines scoring exactly 0 (VADER-neutral) so I can flag the neutral-heavy caveat.
     Return tidy: trading_day, ticker, sector, sent.
   - sector_sentiment_index(scores): forward-fill each ticker's daily score across the full trading
     calendar (carry the last sentiment until new news; neutral 0 before a ticker's first headline),
     then the sector index = equal-weight mean of the sector's tickers' filled scores per trading day.
     Return wide: index = trading_day, columns = the 10 sectors. Document the no-headline choice
     (forward-fill) and why (news sentiment persists until updated; avoids a jumpy index driven by which
     tickers happen to have news that day).

3) In scripts/run_part_b.py, after the funds block: build the scores and the sector index and save
   results/data/sector_sentiment_index.csv (EXACT name). Print the neutral share, the per-sector mean
   sentiment, and the date range.

4) Add results/figures/sentiment_index.png (reuse plotstyle + SECTOR_COLORS): the 10 sector sentiment
   indices over time (a light rolling mean is fine for readability - state the window). Self-contained
   caption.

When done: (a) print the neutral share and per-sector means and confirm the index is defined on every
trading day; (b) name the riskiest bit (I expect the VADER neutral-heavy / false-neutral issue, or
forward-fill staleness) and how to sanity-check it; (c) print a draft log entry with sections 1-3
filled, leave the last two for me.]

## What the assistant produced
- src/sentiment.py: guarded _ensure_vader (with a certifi SSL-context fallback for the macOS "certificate
  verify failed" error), score_headlines (raw-title VADER compound, aggregated to per-(ticker, day) mean,
  neutral share at .attrs['neutral_share']), and sector_sentiment_index (forward-fill per ticker across
  the calendar, 0 before first headline, equal-weight sector means).
- run_sentiment() block in scripts/run_part_b.py and fig_sentiment in scripts/make_fund_figures.py;
  installed nltk from requirements-dev.txt.
- Results: 146,830 headlines scored, 48.8% VADER-neutral (compound == 0); index defined on all 1006
  trading days (2020-01-02 to 2023-12-29); per-sector means all mildly positive in a tight band (+0.096
  to +0.165); sampled compound==0 headlines to evidence false neutrals. Saved
  results/data/sector_sentiment_index.csv (1006x10) and results/figures/sentiment_index.png.

## What was wrong or risky
- VADER is neutral-heavy: 48.8% of headlines score exactly 0, and many are false neutrals - finance-
  directional headlines VADER's social-media lexicon misses (e.g. "...Down On Response To Short", "ARK
  Invest Just Panned", "7 Cheap Oil Stocks to Buy"). This mutes real signal and pulls every sector into
  the same tight mildly-positive band (~+0.10), so the index barely differentiates sectors.
- Forward-fill staleness: a sparse-news ticker carries its last score for days, so the index can reflect
  old news rather than genuine silence.
- The macOS nltk certificate error: the guarded nltk.download failed with "certificate verify failed",
  so a certifi CA-bundle fallback was needed - a deviation from the bare nltk.download in the brief that
  I should mention in the write-up.

## What I changed and why
- Confirmed the 48.8% neutral share matches the brief's "about half neutral" warning, and hand-checked a
  sample of compound==0 headlines to confirm the false-neutral rate is non-trivial.
- Verified the index is defined on every trading day and that the early-2020 COVID dip into negative
  gives it face validity.
- Kept the forward-fill choice (news sentiment persists until updated; avoids a jumpy composition-driven
  index) but noted the staleness limitation for the fusion step.
- Accepted the certifi SSL fallback because the plain nltk.download failed on macOS; it only triggers on
  failure so it is harmless elsewhere, and I will flag it as a deviation in the report.

---

# Prompt log 5 - Station 3 sentiment fusion (baseline tilt)

## What I wanted
- Fold LAGGED equity sentiment into the four equity funds, look-ahead safe (a rebalance at t uses
  sentiment dated strictly before t). In src/fusion.py: apply_sentiment(base_weights, ticker_sentiment,
  lam=0.5) that z-scores lagged sentiment across held names and sets w proportional to base*(1+lam*z),
  clipped long-only and renormalised; plus a fusion path reusing oos_backtest's walk-forward structure to
  build a "<method> + Sentiment" fund beside each equity base. Append the tilted funds to fund_returns.csv,
  fund_weights.csv and performance_metrics.csv; write fusion_comparison.csv (base vs +Sentiment, per
  metric, with deltas) and fusion_before_after.png; state plainly whether the tilt helped and confirm the
  lag directly.

## Prompt(s)
- [Station 3 sentiment fusion (required, baseline tilt). Fold the LAGGED equity sentiment into the equity
funds and report the before-vs-after effect. Follow CLAUDE.md: LOOK-AHEAD SAFE - a rebalance on day t
uses sentiment dated strictly before t (>= 1 trading-day lag).

1) In src/fusion.py implement apply_sentiment(base_weights, ticker_sentiment, lam=0.5):
   - base_weights: a per-ticker weight Series from a portfolios weight function.
   - ticker_sentiment: each held ticker's lagged sentiment as of the last trading day BEFORE the
     rebalance (from score_headlines, forward-filled, shifted >= 1 day).
   - tilt: standardise the lagged sentiment cross-sectionally across the held names (z-score), then
     w_tilted proportional to base_weight * (1 + lam * z), clipped at 0 (long-only), renormalised to
     sum 1. State lam and that a naive tilt may not help.

2) Add a fusion path to the EQUITY backtest reusing oos_backtest's walk-forward structure: at each
   rebalance t, form base weights from returns before t, tilt them with the sentiment dated before t,
   and apply forward. Do this for the four equity methods, producing a "<method> + Sentiment" fund
   beside each base.

3) Save: append the sentiment-tilted equity funds to results/data/fund_returns.csv and
   fund_weights.csv and their metrics to results/tables/performance_metrics.csv; write
   results/tables/fusion_comparison.csv (per equity method: base vs +Sentiment ann_return, ann_vol,
   sharpe, max_drawdown, and the delta); add results/figures/fusion_before_after.png (growth of $1 or a
   Sharpe barplot, base vs +Sentiment), self-contained caption.

When done: (a) print the fusion_comparison table and state plainly whether the tilt helped or hurt
(honest either way); (b) confirm the lag with a direct check (sentiment used at rebalance t is dated
< t); (c) name the riskiest bit and print a draft log entry with sections 1-3 filled, leave the last
two for me.]

## What the assistant produced
- src/fusion.py: apply_sentiment (z-score lagged sentiment across held names, w proportional to
  base*(1+lam*z), clipped long-only, renormalised) and fusion_backtest (reuses
  portfolios._month_end_rebalances and WEIGHT_FUNCS; base weights from returns < t, sentiment dated
  strictly before t, applied forward; returns a (rebalance_date, sentiment_date) lag audit).
- A daily_ticker_sentiment helper in src/sentiment.py (forward-filled per-ticker daily score), a
  run_fusion block in scripts/run_part_b.py, and fig_fusion in scripts/make_fund_figures.py.
- Results (lam=0.5): the tilt raised Sharpe for equal-weight (+0.070) and risk-parity (+0.024) but hurt
  min-variance (-0.280) and max-sharpe (-0.128); mean Sharpe delta -0.078, so on average it did not beat
  the base. Lag check passed (sentiment dated the prior trading day at every rebalance). Invariants
  verified (no-op at lam=0 and under constant sentiment, long-only, sums to 1). Outputs now hold 16 funds
  plus fusion_comparison.csv and fusion_before_after.png.

## What was wrong or risky
- The naive sentiment tilt did not help on average (mean Sharpe delta -0.078): it helped the diversified
  funds (equal-weight +0.070, risk-parity +0.024) but hurt the concentrated optimisers (min-variance
  -0.280, max-sharpe -0.128). For a concentrated fund the z-score is computed over a tiny (~6-13) cross-
  section dominated by VADER's 48.8% neutrals, so a couple of non-neutral names swing the tilt and it
  behaves like noise, raising vol and drawdown. Forward-filled stale sentiment also feeds the tilt. This
  is an honest negative result, explained.
- The Sharpe barplot first choked on the four new +sentiment rows (not in its method map); caught and
  fixed by filtering that figure to the 12 base funds.

## What I changed and why
- Verified the look-ahead lag directly: at every rebalance the sentiment used is dated the prior trading
  day (strictly before t), with a recorded (rebalance_date, sentiment_date) audit; sentiment_date 
  rebalance_date for all.
- Confirmed the apply_sentiment invariants myself (no-op at lam=0 and under constant sentiment, long-
  only, sums to 1, non-held names stay 0).
- Read the result honestly as a clean negative: the naive tilt does not beat the base on average, and I
  report the diversified-vs-concentrated split as the interpretation rather than claim the tilt works.
- Kept lam=0.5 as a modest baseline (a constant at the top of run_part_b.py, easy to sweep) and noted a
  lam sweep (0, 0.25, 0.5, 1.0) as the check that the tilt, not a bug, drives the min-variance drop.

---

# Prompt log 6 - Station 3 innovation (the Noise Tax attention factor)

## What I wanted
- A walk-forward, sector-neutral attention-concentration tilt on the equity funds, look-ahead safe (signal
  at rebalance t uses headline data dated strictly before t), reusing noise_tax and the portfolios/fusion
  walk-forward structure, benchmarked against BOTH the base fund and the required sentiment fusion. In
  noise_tax.py: rolling_attention_concentration(counts, window=252) = trailing burst share from past
  counts only. In src/attention_factor.py: apply_attention (sector-neutral within-sector z-score tilt
  toward LOW concentration, sector totals unchanged) and attention_backtest (mirror of fusion_backtest).
  Append "<method> + Attention" funds, write factor_comparison.csv (Base vs +Sentiment vs +Attention) and
  factor_comparison.png; confirm the lag and sector-neutrality directly.

## Prompt(s)
- [Innovation: the Noise Tax attention factor (Station 3, my 30% extension). Build a walk-forward,
sector-neutral attention-concentration tilt on the equity funds, backtest it out-of-sample, and
benchmark it against the base fund AND the required sentiment fusion. Follow CLAUDE.md: the signal is now
TRADED, so it must be LOOK-AHEAD SAFE - the signal at rebalance t uses only headline data dated strictly
before t (>= 1 trading-day lag). Reuse noise_tax and the walk-forward structure in portfolios/fusion.
Equity-only.

Motivation (put in the docstring): Part A found "loud = riskier" fails market-wide but a weak signal
survives WITHIN sectors via attention concentration. The Old Money thesis ("patience beats hype") tilts
AWAY from loud, concentrated-attention names toward quiet compounders, within sector.

1) In noise_tax.py add a walk-forward signal:
   - rolling_attention_concentration(attention_counts, window=252): for each ticker, over a trailing
     window of daily headline counts, a concentration score = burst_share over the window (share of the
     ticker's headlines on its own abnormal-attention days, where a day is abnormal if its count > the
     ticker's trailing mean + 2*std). Return a per-(date, ticker) concentration computed from PAST counts
     only. Document that higher = louder/more concentrated.

2) New src/attention_factor.py:
   - apply_attention(base_weights, concentration, ticker_sector, lam=0.5): SECTOR-NEUTRAL tilt. Within
     each sector, z-score the lagged concentration across the sector's HELD names, then w_tilted
     proportional to base_weight * (1 - lam * z) (minus: overweight LOW concentration / quiet), clip
     long-only, and renormalise WITHIN each sector so every sector's total weight is unchanged. A sector
     with no usable spread is left untouched.
   - attention_backtest(equity_wide, method, concentration_daily, ticker_sector, lam, warmup=252):
     mirror fusion_backtest - base weights from returns < t, concentration as of the last trading day
     strictly before t, tilt, apply forward. Return daily returns, weights, first_oos, and a
     (rebalance_date, signal_date) lag audit.

3) In scripts/run_part_b.py after the fusion block: build the concentration signal, run the attention
   tilt for the four equity methods, append "<method> + Attention" funds to fund_returns.csv,
   fund_weights.csv, performance_metrics.csv, and write:
   - results/tables/factor_comparison.csv: per equity method, Base vs +Sentiment vs +Attention
     (ann_return, ann_vol, sharpe, max_drawdown).
   - results/figures/factor_comparison.png: Sharpe (or growth of $1) for Base vs +Sentiment vs
     +Attention across the equity methods, self-contained caption.

When done: (a) print factor_comparison and state plainly whether the attention tilt helped, hurt, or was
neutral versus base and versus the sentiment tilt (honest either way); (b) confirm the lag directly
(signal_date < rebalance_date at every rebalance) and that the tilt is sector-neutral (sector totals
unchanged); (c) name the riskiest bit and print a draft log entry with sections 1-3 filled, leave the
last two for me.]

## What the assistant produced
- noise_tax.rolling_attention_concentration (rolling burst share over a trailing 252-day window, past
  counts only, min_periods=252), src/attention_factor.py (apply_attention + attention_backtest), a
  run_attention block in scripts/run_part_b.py, and fig_factor in scripts/make_fund_figures.py.
- Results (lam=0.5): the sector-neutral attention tilt raised Sharpe vs base for 3/4 methods (equal-weight
  +0.016, risk-parity +0.017, max-sharpe +0.036; min-variance -0.001), mean +0.017, and beat the
  sentiment tilt by a mean +0.095 by avoiding its concentrated-fund losses (min-variance 0.612 vs 0.333;
  max-sharpe 0.758 vs 0.594).
- Lag check passed for both signals (prior trading day); sector-neutral drift 2.8e-17; the concentration
  signal has real cross-sectional dispersion (std ~0.075, range 0.10-0.50) and is first defined
  2020-12-30, before the first rebalance. Outputs now hold 20 funds plus factor_comparison.csv and
  factor_comparison.png.

## What was wrong or risky
- The rolling burst-share signal is fragile on sparse news: the trailing mean+2*std threshold and burst
  share can hinge on a handful of high-count days, and the within-sector z-score is then computed across
  only a sector's held names - for concentrated funds (max-sharpe ~6 names) often 1-2 per sector - so the
  tilt barely engages and what it does is noisy. The danger is a silent near-no-op read as "helped/hurt".
- The edge vs base is small (mean +0.017), consistent with Part A: the pooled "loud = riskier" fails, so
  a sector-neutral tilt can only extract a weak within-sector edge. I should not overclaim - the value is
  robustness, not a large alpha.
- The stronger, honest result is vs the naive sentiment tilt (+0.095 mean), because sector-neutrality
  stops the concentration damage the sentiment tilt caused.

## What I changed and why
- Verified look-ahead directly for both signals (signal_date < rebalance_date, the prior trading day) and
  that the rolling signal uses only past counts (window ending at each row, min_periods=252).
- Verified sector-neutrality directly: at the hardest case (min-variance, final rebalance) the max
  sector-total drift is 2.8e-17 (machine epsilon), so the tilt only reallocates within sectors.
- Guarded against a silent no-op: confirmed the concentration signal has real dispersion (std ~0.075) and
  is defined before the first rebalance, so the tilt genuinely engages.
- Read the result honestly as a weak but robust within-sector edge - mildly positive vs base, clearly
  steadier than the sentiment fusion - matching Part A's "signal only survives within sectors" finding,
  and noted a lam sweep {0, 0.25, 0.5, 1} as a further robustness check.

---

# Prompt log 7 - Station 4 the Old Money Streamlit app

## What I wanted
- A deploy-ready streamlit_app.py at the project root that reads ONLY the precomputed results/ CSVs (never
  importing nltk/sentiment/fusion/portfolios or recomputing a backtest), reuses src.plotstyle for the Old
  Money look, and stays light for the free tier. A header explaining Old Money with a Links placeholder,
  then four investor-journey sections: (1) Compare funds - metrics table, risk-return scatter coloured by
  family, growth-of-$1 overlay per family, family filter; (2) Fact sheet - fund picker with growth of $1,
  key metrics, drawdown, and current holdings; (3) Build an allocation - sliders across funds normalised
  to 100% with the blended weighted-sum growth and metrics (clearly NOT a backtest); (4) Analytics - sector
  sentiment index and the Base vs +Sentiment vs +Attention comparison with a plain-English Noise Tax
  explanation and the honest finding.

## Prompt(s)
- [Station 4: the Streamlit app (the investor journey). Build streamlit_app.py at the repo root. It must
read ONLY the precomputed results/ CSVs - NEVER import nltk, sentiment, fusion, portfolios, or recompute
any backtest. Blending precomputed fund returns into an allocation is fine (a weighted sum, not a
backtest). Reuse plotstyle for the Old Money look (apply_style + palette); keep it light for the free tier.

Data it reads (cache with st.cache_data): results/tables/performance_metrics.csv,
results/data/fund_returns.csv, results/data/fund_weights.csv, results/data/sector_sentiment_index.csv,
results/tables/factor_comparison.csv, results/tables/fusion_comparison.csv.

A short header explains Old Money (beginner-first, goal-based multi-asset funds) with a placeholder
"Links" line for the live URL and repo (I fill it at deploy). Then four sections for the investor journey:
1) Compare funds: the performance_metrics table (formatted), a risk-return scatter (ann_vol x, ann_return
   y, coloured by family), and a growth-of-$1 overlay for a chosen family. Let the user filter by family.
2) Fact sheet: a fund picker showing that fund's growth of $1 (cumprod of its precomputed daily returns,
   dropna), key metrics (ann return, vol, Sharpe, max drawdown), a drawdown curve, and current holdings
   (latest rebalance from fund_weights, top holdings as a bar).
3) Build an allocation: sliders to set weights across a few funds (normalise to 100%), then show the
   blended portfolio's growth of $1 and metrics from the weighted sum of the funds' precomputed daily
   returns (clearly NOT a backtest).
4) Analytics: the sector sentiment index over time, and the Base vs +Sentiment vs +Attention comparison
   from factor_comparison.csv, with a short plain-English explanation of the Noise Tax attention factor
   and the honest finding (weak within-sector edge, more robust than the sentiment tilt).

When done: (a) run `streamlit run streamlit_app.py` locally and confirm it loads with no errors; (b) grep
streamlit_app.py for nltk / oos_backtest / score_headlines and confirm NONE appear (app reads precomputed
only); (c) name the riskiest bit and print a draft log entry with sections 1-3 filled, leave the last two
for me.]

## What the assistant produced
- streamlit_app.py (repo root): cached loaders for the six results/ CSVs, four tabs (compare funds, fact
  sheet, build allocation, analytics), and helper functions for growth of $1, drawdown, and weighted-sum
  metrics. The only src import is plotstyle (matplotlib-only).
- Verified: streamlit run boots clean (HTTP 200, no log errors); AppTest runs all four tabs with 0
  exceptions on the default state and across 7 interactive/edge cases; grep confirms no
  nltk/oos_backtest/score_headlines usage (only a docstring mention of the constraint) and no recompute
  calls.

## What was wrong or risky
- Cross-calendar alignment in the allocation blend: fund_returns.csv unions the equity (252-day) and
  crypto (365-day) calendars, so blending a crypto fund with an equity/combined fund via .dropna()
  intersects to the common (equity) dates and drops crypto's weekend rows. The blend then lives on the
  equity calendar and understates crypto's standalone compounding, so a mixed-family blended growth will
  not match the crypto fact sheet. It stays internally consistent (ppy=365 only when every pick is
  crypto, else 252; the caption says "common trading days"), but a user should know the mixed blend is on
  equity days.
- Secondary: matplotlib figures on the free tier can leak memory; handled by plt.close(fig) after every
  st.pyplot.

## What I changed and why
- Verified precompute-only directly: the only src import is plotstyle (matplotlib-only), and grep found
  no nltk/oos_backtest/score_headlines/recompute (the single hit was the docstring naming the
  constraint), so the deployed app cannot run VADER or a backtest.
- Verified it loads: streamlit run returns HTTP 200 with a clean log, and AppTest runs all four tabs with
  0 exceptions on the default state plus 7 interactive/edge cases.
- Accepted the cross-calendar blend behaviour and made it transparent (caption states "common trading
  days" and the 252/365 rule) rather than hiding it, and closed figures after each render for free-tier
  stability.

---

# Prompt log 8 - <task name>

## What I wanted
<the goal in one or two sentences>

## Prompt(s)
<the prompt(s) you gave the assistant>

## What the assistant produced
<a summary, or the code/text it returned>

## What was wrong or risky
<bugs, look-ahead, wrong assumptions, hallucinated APIs - what you found>

## What I changed and why
<your correction, in your own words>
