# 003 — Procedures

End-to-end pipeline. Each phase produces inputs the next consumes; ordering matters.

## Phase A — Data acquisition

`src/data/fetch.py`, `src/data/universe.py`.

1. **Universe build.** Pull current S&P 500 from Wikipedia
   (`get_sp500_constituents`); record `Symbol`, `GICS Sector`, `GICS Sub-Industry`.
2. **Sector filter.** Drop tickers in GICS Sector 40 (Financials) →
   `filter_by_gics_sector`.
3. **Price fetch.** Daily OHLCV via Databento `EQUS.MINI`, schema `ohlcv-1d`,
   `start = 2023-04-01`, `end = today`. Cached to
   `~/data_lake/databento/equities/L0/ohlcv-1d/`.
4. **ADV filter (currently missing — see 007).** Compute
   `(close * volume).rolling(126).mean()` per ticker; drop tickers whose 6-month
   median is below $3M. The function exists (`filter_by_adv`) but `main.py` doesn't
   call it.
5. **Fundamental fetch.** SimFin TTM income, balance, share-prices →
   PE, PB, ROE, GPOA, daily-aligned with a 90-day publication lag to prevent
   look-ahead. Cached to `~/data_lake/simfin/`.

Output: `close`, `volume` (DataFrames; daily × ticker), `fundamentals` (dict of 4
DataFrames; daily × ticker, all aligned to the price index).

## Phase B — Feature engineering

`src/features/factors.py`, `src/features/build.py`.

1. Compute each factor from Phase A on **daily** frequency (not yet weekly).
   Formulas in [`002-requirements.md`](002-requirements.md) §3.
2. **Resample to Wednesday.** `df.resample("W-WED").last()` for every factor.
   Produces a (Wednesday-date × ticker) matrix per factor.
3. **Stack.** Concat factors along columns → MultiIndex (date, ticker) × feature_name.
4. **Build target.** Wednesday close → `pct_change()` → `shift(-1)` for forward
   1-week return → cross-sectional z-score per week.

Output: `features` (MultiIndex (date, ticker) × N_factors), `target` (MultiIndex
(date, ticker) × 1).

## Phase C — Neutralization

`src/features/neutralize.py`. **This step is currently NOT wired into the pipeline
(main.py never calls `full_neutralization`).** Per paper p.13 the order is fixed:

1. **Winsorize 2% / 98%** per factor, cross-sectionally each week.
2. **Iterative cross-sectional z-score, 10 iterations** per factor. Each iteration
   recomputes mean/std after the prior pass, which dampens any remaining outliers.
3. **Cap at ±3** (per paper; not currently implemented).
4. **Subtract industry median.** For each (week, factor), subtract the median value
   among tickers in the same GICS industry. Paper subtracts region × industry; for
   Phase 1 (US-only) these collapse to industry.
5. **Missing-data rule** (per paper p.13; not currently implemented):
   - Drop tickers with > 10 missing factors that week.
   - For tickers kept, replace remaining missing values with 0.

**Why the order matters:** if you industry-neutralize before z-scoring, the
within-industry variance you remove changes the z-score scale and the model
ends up learning sector tilts instead of stock-level alpha.

Output: `features_neutralized` (same shape as `features`, but each factor column
is mean ≈ 0 / std ≈ 1 within each industry-week cell).

## Phase D — Rolling-window training

`src/model/train.py::rolling_train_predict`.

For `i = train_weeks + val_weeks … T`:

```
train: dates[i - train_weeks - val_weeks : i - val_weeks]
val:   dates[i - val_weeks : i]
test:  dates[i] (single Wednesday)
```

- Retrain only every `retrain_freq = 12` weeks; reuse the existing booster between
  retrains (predictions still recomputed each week).
- Use early stopping on the validation set (50 rounds).
- `xgb.DMatrix` for both train and val; predictions are continuous scores.

Output: `predictions` (Series indexed by (date, ticker), real-valued).

## Phase E — Portfolio construction

`src/backtest/portfolio.py`.

1. **Quintile assignment.** `pd.qcut(predictions, q=5)` per week →
   quintile labels 1–5.
2. **Long Q5, short Q1**, equal-weight within each leg.
   - Phase 2: form quintiles within (region × industry) buckets.
3. **ADV scaling** (currently missing — see 007). For tickers below the $20M ADV
   threshold, scale weight proportionally to (ADV / $20M).
4. **Compute returns.** `long_ret = mean(week_ret[long])`, similarly for short;
   `ls_ret = long_ret − short_ret`.
5. **Compute turnover** as the symmetric difference between this week's
   long/short baskets and last week's, normalized to [0, 1] per leg.
6. **Apply costs.** `ls_ret_net = ls_ret − turnover * 1.5bps * 2` (both sides).
7. **Apply execution lag** (currently only used in alpha-decay diagnostic, not in
   main pipeline). Returns should be measured from Thursday close → Thursday close
   when running with 1-day lag.

Output: `portfolio` (DataFrame indexed by Wed-date with columns `long_return`,
`short_return`, `ls_return`, `ls_return_net`, `turnover`).

## Phase F — Diagnostics

`src/diagnostics/shap_analysis.py`.

1. **SHAP** on the most recent trained booster: beeswarm summary plot + mean-|SHAP|
   bar chart. Validation: R1W should be top, UPDOWN1W second (Phase 2).
2. **Alpha decay**: re-run Phase E with `execution_lag_days ∈ {0, 1, 2, 3, 4}`,
   record IR. Expect monotone decay (paper p.7: 14.9% → 6.8% over 0–4 days).
3. **Weekday effect** (currently `NotImplementedError`): rebuild features and
   target keyed off Mon/Tue/Wed/Thu/Fri rather than only Wednesday; rerun the
   model and backtest for each. Expect Thursday > Monday (paper p.8).
4. **Quintile spread**: per-quintile annualized return; should be monotonic.
5. **Crisis-period table** and **performance-by-market-quintile** (paper pp.11–12)
   require Phase 2 history.

Output: PNGs into `notebooks/figures/`, metric tables into `data/processed/`.

## Pipeline orchestration

`src/main.py::run_pipeline` runs A → B → D → E (Phase C neutralization is **not
called**, see 007). Diagnostics are not invoked from `main.py`; run
`src.diagnostics.shap_analysis` separately or via the Streamlit app
([`006-streamlit-app.md`](006-streamlit-app.md)).
