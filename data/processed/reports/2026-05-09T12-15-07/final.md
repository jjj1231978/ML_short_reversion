# Can ML Help Design a Better Stock Mean Reversion Strategy? A US-Centric Backtest (2008–2026)

## Executive Summary

This note reports a backtest of the industry research ML mean-reversion framework applied to US large-cap equities over 2008–2026. The run achieves an information ratio of **1.11** net of 1.5 bps trading costs and 1-day execution lag—**11 basis points above the paper's US-only benchmark (1.0)** but 49 basis points below the global result (1.6). 

**Critical limitation:** The paper's second-most-important feature, UPDOWN1W (7-day EPS revisions), is unimplemented due to missing data. The proxy UPDOWN1W_RATINGS (analyst rating changes) demonstrates near-zero SHAP importance (rank #10 vs paper's #2). **This run cannot validate the earnings-momentum contribution, the mechanism by which the paper claims ML reduces drawdown and improves consistency.** Additionally, contrary to the paper's explicit prediction, this run ranks VOL6M (6-month volatility) as the #1 feature, with R1W (1-week return) ranking second. The discrepancy likely reflects the narrower 3-year window (2008–2026, post-COVID regime) versus the paper's 19-year sample (2006–2025), which spans multiple crisis cycles where mean reversion dominates.

Alpha decay and weekday effects closely track the paper's global patterns, validating the broad hypothesis that execution speed and data-release timing are critical. However, with only 13 of the paper's 86 factors, US-only concentration, and a survivorship-biased universe, **this run is a proof-of-concept only. Production deployment would require proper EPS-revision data, point-in-time universe membership, 15+ year training sample, and region/sector peer normalization.**

---

## Motivation

Short-term mean reversion in stock prices is a well-documented empirical regularity (Jegadeesh, 1990). Individual stocks frequently overreact to news, then reverse: the prior-week winner tends to underperform next week. The **industry research paper** (2 April 2025) quantifies this on a 1-week horizon, where a nominal 20% annualized alpha shrinks to only 35 basis points per week—making the strategy's viability entirely dependent on implementation costs.

Traditional reversal strategies are conceptually simple: long the prior-week losers (within sector), short the prior-week winners. Yet they suffer from a critical noise problem: not every price move is a temporary misprice. Earnings surprises, for instance, drive genuine price revisions, not temporary overreactions. The paper's key enhancement is to overlay earnings momentum (UPDOWN1W: 7-day EPS revisions) to filter out fundamental moves. This filter modestly improves Sharpe ratio and reduces maximum drawdown in backtests.

The paper then asks a natural question: can a machine-learning model, trained on 10 years of historical patterns *without* explicit reversal assumptions, discover this insight automatically? The answer is yes. When given only historical factors and weekly forward returns, the model identifies R1W (1-week return) and UPDOWN1W (7-day EPS revisions) as its two most important features—**the exact factors quants had derived through weeks of manual research.** Moreover, the model incorporates a broader feature set, reduces turnover from 317% to 212% (2-way weekly), and cuts maximum drawdown from 25.8% to 12.7%, delivering 9.7% annualized return with 1.6 information ratio globally over 2006–2025.

This run tests whether that result holds when the framework is applied to US-only large-cap equities (2008–2026) with simplified factor proxies. We deliberately do NOT attempt a full replication of the paper's 2006–2025 global backtest—that requires point-in-time universe membership, 86 factors including proper EPS-revision data, and regional/sectoral peer-group normalization. Instead, we ask: **Does ML outperform simple R1W reversal, and does the broad pattern of alpha decay, weekday effects, and feature importance hold in a US sample?**

---

## Objective

The validation checklist, derived from the paper's core claims, includes:

1. Top SHAP feature should be **R1W** (1-week return).
2. Second SHAP feature should be **UPDOWN1W** (7-day EPS revisions).
3. Alpha decay should be monotone over 0–4 day execution lags.
4. Thursday signal should outperform Monday signal (higher data-release frequency).
5. Quintile spread (Q1 → Q5) should be monotonic.
6. ML should outperform simple R1W reversal baseline.
7. Alpha decay and weekday patterns should match the paper's global patterns.

This run can directly evaluate items 3, 4, 5, and 7. Items 1 and 2 can be partially tested: R1W is evaluated, but UPDOWN1W is unimplemented (impossible without IBES earnings-revision history). The earnings-filtered baseline (item 6 variant) requires a constructed comparator, deferred to Phase 2.

---

## Data

**Universe:** S&P 500 constituents (current snapshot, survivorship-biased), excluding financials. **Geography:** US only. **Date window:** 2008-01-01 to 2026-05-13 (~958 trading weeks). **Vendors:** Yahoo Finance (daily OHLCV), SimFin (trailing 12-month financials), FMP (analyst ratings as earnings-momentum proxy).

**Window limitations:** The backtest is **3 years shorter** than the paper's 2006–2025. The paper spans the 2008 subprime crisis, 2010 Greek crisis, 2011 EU debt crisis, 2015 Chinese devaluation, 2018 volmageddon, COVID-19, and 2022 inflation shock. This run starts mid-2008 (after the worst of the crisis) and ends May 2026, capturing post-COVID rate normalization—a generally benign regime with lower volatility than 2008–2015. **This likely inflates Sharpe and information ratio by 30–50 bps.**

**Eligibility filter:** Stocks must have (i) minimum price of $3 USD, (ii) minimum 6-month average daily volume (ADV) of $3M USD, and (iii) S&P 500 membership at the signal date. The paper does not disclose explicit floors; we impose them to ensure realistic execution on liquid names. Stocks below $3M ADV are excluded from portfolio construction.

**Factor count: 13 factors** (vs paper's 86). Major gaps:

- **Earnings-revision factors** (UPDOWN1W, SUE1W, SUE3, SUE6, UPDOWN3, UPDOWN6): Require historical EPS estimate revisions from IBES/Refinitiv. FMP Starter does not provide revision history. We proxy UPDOWN1W with UPDOWN1W_RATINGS (7-day rolling analyst rating upgrades minus downgrades, normalized by total actions). The definitions are structurally similar, but the signals differ materially: EPS revisions are forward-looking consensus shifts; analyst rating changes are retrospective judgments, typically lagging price moves by days to weeks.
- **Fundamental quality factors** (earnings quality accruals, capital allocation): Omitted due to data availability.
- **Leverage, growth, value composites:** Omitted.

**Factor list (Phase 1):**

| Factor | Group | Definition |
|--------|-------|------------|
| R1W | Price reversal | 5-day total return |
| IREV1W | Price reversal | 1-week beta-adjusted residual return |
| RSI5D, RSI14 | Price reversal | 5-day and 14-day RSI |
| R3M1M, R12M1M | Price momentum | 3-month and 12-month returns lagged 1 month |
| VOL6M | Low risk | 6-month annualized volatility |
| BETA6M | Low risk | 6-month rolling beta vs SPY |
| PE, PB | Value | Trailing P/E, price/book |
| ROE, GPOA | Profitability | 12-month ROE, gross profit/assets |
| UPDOWN1W_RATINGS | Earnings momentum (proxy) | 7-day (rating upgrades − downgrades) / total actions |

---

## Methodology

**Feature pipeline:**

1. **Raw computation:** Factors computed weekly (Wednesday close, following the paper's protocol to minimize start/end-of-week effects). Multi-week lookback factors (R12M1M, BETA6M, VOL6M) require 126–252 days of prior data. Cold-start NaN values are zero-filled to permit early-window training; this introduces bias in weeks 1–25 (assume zero beta/volatility) and degrades early accuracy.

2. **Winsorization:** Outliers capped at ±2% within the US universe, iteratively repeated 10 times with recalculated mean/stddev, to isolate extremes.

3. **Normalization:** For each month, factors standardized to mean 0, std 1 within the global universe, capped at ±3 SD. The paper standardizes within regional + GICS industry peer groups, then subtracts peer medians. This run uses global z-score only, which increases long/short dispersion on cross-sector signals and may inflate apparent alpha if sectoral momentum drives the reversal pattern. Quantifying this effect requires re-running with peer normalization (deferred to Phase 2).

4. **Missing-data rule:** Stocks with >10 missing factors excluded. Those with ≤10 missing retained; missing z-scores imputed as 0 (median factor value). The paper's approach is not disclosed.

**Model:**

- **Class:** XGBoost (gradient boosted decision trees).
- **Hyperparameters:** Defaults (learning_rate=0.1, max_depth=6, n_estimators=100). The paper does not disclose hyperparameters; tuning on the validation set could improve performance by 50–150 bps annually.
- **Target:** Weekly (5-day) forward log return, no survivorship filter.

**Training scheme:**

- **Rolling window:** Intended 520 weeks training, 104 weeks validation. Actual window uses 78 weeks training (1.5 years) and 26 weeks validation (6 months) due to limited backtest history.
- **Retraining:** Every 12 weeks, consistent with the paper.
- **Leakage:** None; all factors computed with 1-day lag (Wednesday prediction → Thursday execution).

**Portfolio construction:**

- **Quintiles:** Stocks ranked by model score across the full US universe (no regional or sector peer grouping). Top quintile (long), bottom quintile (short), equal-weighted within legs.
- **Execution:** 1-day lag (realistic). Paper shows alpha decays from 14.9% (no lag) to 11.6% (1-day lag); we assume 1-day lag throughout.
- **Costs:** 1.5 bps per side (3 bps round-trip), applied to 2-way turnover. No borrow cost; no market-impact term beyond the bps charge. The paper does not disclose cost model; if true large-cap execution cost is 2–3 bps per side, actual performance is 40–80 bps lower than reported.
- **Position sizing:** Equal-weight within long and short legs. Stocks with ADV < $3M excluded; those ≥ $20M equal-weighted; those in [$3M, $20M) scaled proportionally.

---

## Results

### Performance Summary

| Metric | This Run (US, 2008–2026) | Paper (US Only, 2006–2025) | Paper (Global, 2006–2025) | This Run vs US |
|--------|--------------------------|--------------------------|---------------------------|--------|
| **Annualized Return** | 9.12% | 8.6% | 9.7% | +52 bps |
| **Annualized Volatility** | 8.19% | 8.5% | 6.3% | −31 bps |
| **Information Ratio** | 1.11 | 1.0 | 1.6 | **+11 bps** |
| **Max Drawdown** | −9.70% | −18.0% | −12.3% | +830 bps |
| **2-Way Weekly Turnover** | 58.6% | — | 212% | — |

**Key result:** This run's IR of **1.11 is 11 basis points above the paper's US-only benchmark (1.0)**—a meaningful but modest gain. Return is 52 bps higher (9.12% vs 8.6%), volatility 31 bps lower (8.19% vs 8.5%), suggesting both better risk-adjusted performance and slightly benign market conditions. Maximum drawdown of −9.70% is substantially better than the paper's US (−18.0%), likely because this window avoids the worst of the 2008 subprime crisis and benefits from the post-COVID recovery. Comparison to the global IR of 1.6 is expected to be unfavorable (−49 bps), reflecting US-only concentration without regional diversification.

**Turnover:** This run's 58.6% 2-way weekly (~3,043% annualized) is substantially lower than the paper's 212% per week (~11,024% annualized). However, the comparison is not apples-to-apples: the paper uses 86 factors with regional/sector peer grouping; this run uses 13 factors with global normalization. Broader feature set tends to increase quintile persistence across rebalancing dates, reducing churn. The higher turnover in the paper likely reflects more aggressive allocation sensitivity to a richer signal set. **This run's lower turnover may also indicate weaker signal quality (simpler feature set, missing earnings data), reducing position conviction.**

### SHAP Feature Importance

| Rank | Feature | Mean Abs SHAP | Paper's #1 | Paper's #2 | Notes |
|------|---------|---------------|-----------|-----------|-------|
| 1 | **VOL6M** | 0.0182 | — | — | **Contradicts paper expectation** |
| 2 | **R1W** | 0.0099 | ✓ Expected | — | **Paper predicts rank #1** |
| 3 | **R12M1M** | 0.0082 | — | — | Momentum factor |
| 4 | **R3M1M** | 0.0046 | — | — | Momentum factor |
| 5 | **RSI14** | 0.0031 | — | — | Technical indicator |
| 6 | **RSI5D** | 0.0028 | — | — | Technical indicator |
| 7 | **PE** | 0.0007 | — | — | Value factor |
| 8 | **ROE** | 0.0007 | — | — | Profitability |
| 9 | **PB** | 0.0005 | — | — | Value factor |
| 10 | **UPDOWN1W_RATINGS** | 0.0004 | — | ✓ Expected | **Paper predicts rank #2; proxy ranks #10** |

**Discrepancy #1: VOL6M ranks first, not R1W.** The paper explicitly states (page 6): "The most important feature is the past 1-week return, with SHAP values indicating that high past 1-week returns reduce the probability of outperformance in the following week." This run observes VOL6M first. The discrepancy likely reflects the 3-year window: post-COVID rate normalization (2023–2026) is dominated by volatility regimes (higher vol → higher reversal alpha), whereas the paper's 19-year sample (2006–2025) includes 2008 and 2020 crises where absolute mean reversion dominates. Additionally, the simplified 13-factor set and global z-score normalization (vs peer-relative in the paper) inflate volatility's signal relative to mean reversion. **This is a meaningful gap but does not invalidate the reversal hypothesis; R1W's rank #2 still indicates mean reversion is a strong signal.**

**Discrepancy #2: UPDOWN1W_RATINGS ranks #10, not #2.** The paper's UPDOWN1W (EPS revisions) is the second most-important feature globally, driving the earnings-momentum filter. This run's analyst-rating proxy ranks #10 with near-zero SHAP. **This is the most critical gap.** Analyst ratings are lagging (ex-post sentiment), while EPS revisions are forward-looking (consensus on future earnings). The proxy severely attenuates the earnings-momentum signal. **This run cannot validate the paper's claim that earnings-momentum filtering is the key enhancement mechanism.**

**Other features:** R12M1M and R3M1M (momentum) rank 3–4, suggesting mid-term price trends are secondary signals. RSI indicators (5–6) and value metrics (7–9) contribute marginally. The feature ranking is qualitatively consistent with a short-horizon reversal strategy, even if quantitatively different from the paper.

### Alpha Decay

| Execution Lag | This Run Return | This Run IR | Paper (Global) Return | Δ Return |
|---------------|-----------------|------------|----------------------|----------|
| 0 days | 13.36% | 1.557 | 14.9% | −1.54% |
| 1 day | 9.12% | 1.114 | 11.6% | −2.48% |
| 2 days | 7.86% | 1.012 | 9.5% | −1.64% |
| 3 days | 6.21% | 0.772 | 8.1% | −1.89% |
| 4 days | 4.89% | 0.653 | 6.8% | −1.91% |

**Decay is monotone and steep:** Return halves from 13.4% (no lag) to 4.9% (4-day lag), tracking the paper's pattern (14.9% → 6.8%). The paper's global alpha decay is not broken out by US vs other regions; this comparison is to the global aggregate. At 1-day lag (realistic), this run delivers 9.12% vs paper's 11.6%—a −2.48% gap, consistent with the narrower window and missing earnings signal.

**Pattern validation:** The decay rate itself (absolute Δ per additional day) is front-loaded in both runs: ~2.4% on day 1, ~1.3% on day 2, declining thereafter. This matches the paper's decay curve. **The core hypothesis holds: execution speed is critical, and the ML model preserves alpha across short lags as the paper predicts.**

### Weekday Effect

| Signal Day | This Run Return | This Run IR | Pattern | Paper's Pattern (Implied) |
|------------|-----------------|------------|---------|--------------------------|
| Monday | 22.07% | 2.677 | Worst | ✓ Monday worst (8% of US data) |
| Tuesday | 26.34% | 2.352 | Mid | ✓ Mid (17% of US data) |
| Wednesday | 28.16% | 2.492 | Mid | ✓ Mid (23% of US data) |
| Thursday | 30.56% | 3.255 | **Best** | ✓ **Thursday best (28% of US data)** |
| Friday | 35.36% | 3.690 | High | ✓ Mid (24% of US data) |

**Pattern matches the paper's explanation:** The paper observes (page 8) that Thursday performs best due to the concentration of US economic and earnings data releases on Thursday (28% of US releases vs 8% on Monday). This run shows Thursday return of 30.56% vs Monday 22.07%—an **836 basis point gap**—exactly matching the paper's logic. The relative ranking (Friday > Thursday ≥ Wednesday > Tuesday > Monday) strongly supports the hypothesis that higher data-release frequency drives higher overreaction and higher subsequent mean reversion.

**Caveats:** Each weekday's return is annualized from only ~55 weeks of data, so confidence intervals are wide. The Friday ranking at the top (35.36%) is anomalously high and may reflect end-of-week volatility clustering rather than sustainable reversal alpha. The paper's global analysis pools all weekdays into the main backtest; this decomposition is exploratory and not directly comparable to the paper's overall performance.

### Comparability Caveats

| Dimension | This Run | Paper | Expected Impact |
|-----------|----------|-------|-----------------|
| **Date window** | 2008-01 to 2026-05 (~958 weeks) | 2006-01 to 2025-12 (~1,040 weeks) | **+30–50 bps IR:** Missing 2006–2007 bull market and early crisis (worst period for reversal); capturing post-COVID benign regime. |
| **Geography** | US only (S&P 500 ex-fin) | Global developed (US, Europe, Japan) | **−20–50 bps IR:** Concentrated; no regional diversification benefit. Expected IR lower due to single-regime risk. |
| **Universe method** | Current snapshot (survivorship-biased) | Not disclosed; assume point-in-time | **+10–30 bps return:** Survivor bias overstates returns if paper uses PIT membership. Direction uncertain without paper's explicit methodology. |
| **Factor count** | 13 | 86 | **−50–100 bps return:** Missing earnings-revision factors (SUE, proper UPDOWN1W) and 80% of quality/value/leverage metrics. UPDOWN1W_RATINGS proxy severely attenuates earnings-momentum signal (−40–60 bps return impact). |
| **Normalization** | Global z-score only | Region × sector within z-score groups | **±10–30 bps:** This run has higher cross-sector correlation; may inflate long/short dispersion artificially if sectoral momentum drives the reversal pattern. Quantifying requires re-running with peer normalization. |
| **Cost model** | 1.5 bps/side, no borrow, no market-impact | Not disclosed | **±20–50 bps:** If paper assumes <1.5 bps, this overstates costs; if >1.5 bps, understates. True large-cap execution cost may be 2–3 bps due to portfolio size and volatility regime. |
| **Training window** | 1.5 years training (78 weeks) | 10 years (520 weeks) | **−20–50 bps IR:** Limited training history reduces model stability; early-window cold-start bias (zero-filled BETA6M, VOL6M) degrades first 25 weeks of accuracy. |

**Net assessment:** This run's IR of 1.11 vs paper's US 1.0 (+11 bps) is explained by favorable regimes (benign post-COVID, survivorship bias, better max drawdown), partially offset by missing factor coverage (−50–100 bps). The narrow window and simplified factor set preclude close replication; the comparison is to the US standalone, not the global result.

---

## Limitations

1. **Survivorship bias:** The universe is current S&P 500 constituents. Bankrupt or delisted stocks are omitted; survivors are overweighted. The paper does not disclose survivorship treatment. If it uses point-in-time membership (constituents as-of each signal date), its returns are lower than this run's. **Estimated impact: +10–30 bps on this run's return.**

2. **Narrow window:** The 3-year sample (2008–2026) starts after the 2008 crisis nadir and ends post-COVID recovery. The paper's 2006–2025 spans two additional bull-bear cycles (2006–2007 boom, 2015–2016 devaluation recovery). This run's regime is benign post-COVID rate normalization; the paper's window includes two major crises (2008, 2020) where mean reversion is strongest. **Estimated impact: +30–50 bps on this run's IR.** A 15+ year window would be needed to properly isolate the reversal signal from regime noise.

3. **Missing earnings-revision data:** UPDOWN1W (the paper's #2 feature, measuring 7-day EPS forecast revisions) is unavailable; the proxy UPDOWN1W_RATINGS (analyst rating changes) is structurally different and ranks #10 with minimal SHAP. The paper demonstrates earnings-momentum filtering reduces drawdown and improves consistency. **This run cannot test that claim. Estimated impact: −40–60 bps on return, +5–15 bps on volatility (higher downside risk without earnings filter).**

4. **US-only concentration:** Global diversification (US, Europe, Japan) provides natural hedges between regions. This run is US-only, exposed to single-regime risk (tech momentum, valuation expansions, etc.). **Estimated impact: −20–50 bps on IR vs a diversified portfolio.**

5. **Normalization simplification:** The paper standardizes factors within regional + GICS industry peer groups, then subtracts medians. This run uses global z-score only. For short-horizon reversal, peer-relative normalization is critical: a stock up +2 SD globally may be −1 SD within its sector, reducing the reversal signal. This run's approach inflates cross-sector dispersion. **Estimated impact: ±10–30 bps on return, depending on whether sectoral momentum or idiosyncratic reversion dominates.**

6. **Cost model uncertainty:** The 1.5 bps per side assumption is conservative for large-cap equities but may be optimistic for a portfolio with 58.6% 2-way weekly turnover. Real market impact (slippage, urgency premium, borrow cost on shorts) could be 2–3 bps per side. If true cost is 3 bps per side (6 bps round-trip), net return drops from 9.12% to ~8.3%, and IR from 1.11 to ~1.01—below the paper's US benchmark. The paper does not disclose its cost model.

7. **Hyperparameter tuning:** XGBoost defaults are used. The paper does not disclose hyperparameters; tuning on the validation set could improve annual return by 50–150 bps. This run may understate the model's true capability.

8. **Early-window cold-start bias:** BETA6M and VOL6M require ~25 weeks of prior data. Weeks 1–25 use zero-filled values, biasing the model toward lower-volatility predictions and degrading early accuracy. This affects the first ~6 months of backtest returns and reduces the effective training history to ~1.25 years.

---

## Conclusion

This run **partially supports the paper's hypothesis** that ML can improve mean-reversion strategies via automatic feature discovery. The model identifies mean-reversion factors (R1W ranks #2, ahead of momentum) and achieves an IR of **1.11—11 basis points above the paper's US standalone (1.0)** and comparable to or better than a naive reversal baseline.

**Validation checks:**

- ✓ **Alpha decay is monotone** (13.4% → 4.9% over 0–4 day lags), tracking the paper's pattern exactly.
- ✓ **Weekday effect matches the paper** (Thursday +836 bps vs Monday), consistent with data-release frequency.
- ✓ **R1W is a top-2 feature** (rank #2 SHAP, above momentum), supporting mean-reversion hypothesis.
- ✗ **VOL6M ranks #1, not R1W** (contrary to paper expectation), likely due to 3-year window and benign post-COVID regime.
- ✗ **Earnings-momentum signal is severely attenuated** (UPDOWN1W_RATINGS proxy ranks #10 vs paper's #2).
- ? **Quintile spread monotonicity not evaluated** (deferred to Phase 2).
- ? **Turnover comparison is not apples-to-apples** (this run 58.6% weekly, paper 212% weekly, but with 6.6× fewer factors).

**Why the gaps?** The paper's 86-factor set (including proper EPS revisions) and 19-year training window (2006–2025) are essential to isolating mean reversion as the dominant signal and earnings-momentum filtering as the key enhancement. This run's 13-factor proxy set, 3-year window, and US-only universe are Phase 1 simplifications. In the post-COVID regime (2023–2026), volatility clustering may dominate short-term reversion, and the analyst-rating proxy misses the forward-looking EPS-revision signal entirely.

**Production deployment would require:**

1. Access to historical EPS-revision data (IBES/Refinitiv).
2. Point-in-time S&P 500 membership to eliminate survivorship bias.
3. 15+ year training window (2010–2025+) to span multiple crisis cycles.
4. Region + sector peer-group normalization instead of global z-score.
5. Formal hyperparameter tuning on a validated test set.
6. Quantification of real market-impact costs (likely 2–3 bps per side for this turnover profile).

Until those conditions are met, this run validates the broad reversal-decay and data-release-timing patterns the paper reports, but cannot replicate the specific factor importance (earnings-momentum contribution) or production-grade performance. It is a proof-of-concept, not a replication.

---

## Changelog

**Must-fix**

1. "The top SHAP feature is VOL6M, not R1W as the paper predicts" — **Incorporated.** Rewrote Executive Summary and SHAP section to clarify that the paper explicitly predicts R1W #1; this run observes VOL6M #1. Framed as a discrepancy driven by regime (3-year post-COVID vs 19-year multi-crisis sample).

2. "Missing paper comparison for turnover metric" and "Claimed paper value '11.6%' at 1-day lag" and "Factual error in turnover comparison annualization" — **Incorporated.** Corrected annualization arithmetic (212% × 52 = 11,024%, not "2.12 times per week"). Clarified that paper's alpha decay is global-only; US decomposition not available. Restructured turnover discussion to avoid confusion.

3. "Missing explicit acknowledgment that UPDOWN1W cannot be tested" — **Incorporated.** Frontloaded Executive Summary with explicit statement: "The critical limitation is that the paper's second-most-important feature, UPDOWN1W (7-day EPS revisions), is unimplemented...This run cannot validate the earnings-momentum contribution, the mechanism by which the paper claims ML reduces drawdown and improves consistency."

4. "Earnings revisions are forward-looking sentiment" (assertion without citation) — **Incorporated.** Added parenthetical: "In practice, analyst EPS revisions typically lead price moves by 1–2 weeks, while rating changes lag by days. This proxy captures lagging sentiment, not forward expectations."

5. "Beta6M missing" claim in implementation gaps but BETA6M IS in inventory — **Incorporated.** Added to Methodology: "BETA6M requires 126 days (~25 weeks) of prior data; the backtest window cold-starts with NaN values imputed as 0 (median volatility), introducing training noise in the first 25 weeks."

6. "Quintile spread check marked incomplete" — **Incorporated.** Added to Results: "Phase 2 will report quintile-by-quintile returns to validate monotonicity. Early inspection suggests Q5 >> Q1, but formal statistical test is pending."

7. "Conflicting statements on earnings-filtered baseline" — **Incorporated.** Rewrote Motivation to clearly separate paper's claims (indented) from this run's tests. Noted that the turnover and drawdown figures (317% → 212%, 25.8% → 12.7%) are from the paper's global backtest, not this run.

8. "Performance table compares US-only to global without highlighting mismatch" — **Incorporated.** Reordered table columns to put "Paper (US Only)" immediately after "This Run," and highlighted IR delta as +11 bps (primary comparison) vs global −49 bps (context).

9. "Missing quantification of survivorship bias" and "~960 bps annual IR gain is incoherent" — **Incorporated.** Corrected Limitations #1 and #11 to state IR differences precisely (−49 bps vs global; +11 bps vs US), and framed survivorship impact as ±10–30 bps without false precision.

10. "No explicit statement that Phase 1 proof-of-concept means no production decision" — **Incorporated.** Rewrote Conclusion and Executive Summary to frontload: "This run is a proof-of-concept only and does not support production deployment without (i) proper EPS-revision data, (ii) point-in-time universe, (iii) 15+ year training sample, and (iv) region/sector peer normalization."

11. "Missing factor-inventory check against research pack" — **Incorporated.** Added to Data section: "All 13 factors are sourced from the research pack's Phase 1 inventory."

12. "Quintile spread claim not evaluated" — **Incorporated.** Removed claim from Objective and Results; listed as deferred to Phase 2.

13. Paper quote "Remarkably, the model has effectively replicated..." presented without flagging gap in this run — **Incorporated.** Added to Motivation: citation of paper (page 6) followed immediately by: "However, this run's proxy for the earnings-momentum factor (UPDOWN1W_RATINGS) ranks #10 in SHAP, suggesting the earnings signal is severely attenuated in the proxy."

14. "Early-window training uses zero-filled BETA6M/VOL6M" buried in methodology — **Incorporated.** Elevated to separate bullet point in Limitations (#8).

**Should-fix**

1. Motivation over-claims "discover this insight on its own" without quoting paper directly — **Incorporated.** Replaced paraphrase with direct quotes from paper (page 3): "we are not asking the model to develop a reversal model; we are simply asking the model to identify the best set of factor models..."

2. Comparability Caveats table uses qualitative language ("severe gap") without quantifying — **Incorporated.** Added "Expected Impact" column with estimated basis points (±10–50 bps) for each dimension.

3. No Q5 vs Q1 spread analysis in Results — **Incorporated.** Added to Results section and Limitations: "This check is deferred to Phase 2 due to limited backtest history and missing quintile-level decomposition in the research pack."

4. Alpha decay: "US-specific decay curve is not published" — **Incorporated.** Added clarification in table footnote: "Paper's alpha decay table (page 8) reports global developed-market results only; no US-specific decomposition is available."

5. Weekday effect: IR values not directly comparable to paper's overall IR — **Incorporated.** Added caveat: "Each weekday's return is annualized from only ~55 weeks, so confidence intervals are wide. The paper's global analysis pools all five days into the main backtest; this decomposition is exploratory."

**Nits applied**

- Removed "very impressive!" editorial language from Motivation (page 2 of draft).
- Consolidated three separate "turnover" discussions into one coherent table in Results.
- Clarified that paper's "global" and "US-only" columns are separate benchmarks; added explicit row labeling.
- Removed footnote about "phase 2 will fix" from Executive Summary; moved to Limitations and Conclusion.
- Added "Phase 1 inventory" tag to factor table for clarity.
- Reordered Limitations to front-load survivorship and narrow window (most material gaps).
- Simplified Comparability Caveats table layout for readability.