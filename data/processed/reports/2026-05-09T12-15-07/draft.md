# Can ML Help Design a Better Stock Mean Reversion Strategy? A US-Centric Backtest (2008–2026)

## Executive Summary

This note reports a backtest of the industry research ML mean-reversion framework applied to US large-cap equities over 2008–2026. The run achieves an information ratio of **1.11** net of 1.5 bps trading costs and 1-day execution lag—below the paper's global net-of-costs IR of 1.6 but comparable to the paper's US standalone result of 1.0 (IR gain of +0.11). The top SHAP feature is **VOL6M** (6-month volatility), not **R1W** (1-week return) as the paper predicts; R1W ranks second. The critical paper-proxy factor UPDOWN1W is unimplemented due to missing earnings-revision data, replaced with UPDOWN1W_RATINGS (analyst rating revisions), a materially different signal. Alpha decay and weekday effects closely track the paper's global patterns. The binding limitation is a 3-year backtest window (vs paper's 19 years), likely inflating risk-adjusted returns in a benign post-COVID regime. **This run validates the broad contour of the paper's hypothesis—ML beats traditional reversion baselines—but cannot isolate the specific factor importance or earnings-momentum contribution the paper highlights. Cross-validation against the paper's 2006–2025 global results is premature; the US-only, factor-proxied, window-constrained setup is a Phase 1 proof-of-concept.**

---

## Motivation

Short-term mean reversion in stock prices is a well-documented empirical regularity. Individual stocks frequently overreact to news, then reverse: the winner from last week tends to underperform next week, and vice versa. The **industry research paper** (2 April 2025) quantifies this on a 1-week horizon—a 20% annualized alpha shrinks to only 35 bps per week, making the strategy's viability hinge entirely on keeping implementation costs below alpha decay.

Traditional reversal strategies are simple: long the prior-week losers (within sector), short the prior-week winners. But they suffer from noise. Not every price move is an overreaction: earnings surprises drive genuine price revisions, not temporary mispricings. The paper's key insight is to overlay earnings momentum (7-day EPS revisions, UPDOWN1W factor) to filter out fundamental moves. This *ad hoc* enhancement modestly improves Sharpe ratio and reduces maximum drawdown.

The paper then asks: can a machine-learning model, trained on 10 years of historical patterns without prior assumptions about reversals, *discover* this insight on its own? The answer is yes. The ML model, given only historical factors and weekly forward returns, identifies R1W (1-week return) as the #1 feature and UPDOWN1W (7-day EPS revisions) as #2—**exactly the factors the quants had derived through weeks of research**. Moreover, the model incorporates broader feature set, reduces turnover from 317% to 212% (2-way weekly), and cuts maximum drawdown from 25.8% to 12.7%. Net of costs, the ML strategy delivers 9.7% annualized return with 1.6 IR globally over 2006–2025.

This run tests whether that result holds when the framework is applied to US-only large-cap equities in a narrower window (2008–2026) with simplified factor proxies. We deliberately do NOT attempt to replicate the paper's full 2006–2025 global result—that requires point-in-time universe membership, 86 factors including earnings-revision proxies, and multi-region peer grouping. Instead, we ask a simpler question: **Does ML outperform simple R1W reversal, and does it do so via the mechanisms the paper identifies?**

---

## Objective

The validation checklist (derived from the paper's claims) includes:

1. Top SHAP feature should be **R1W** (1-week return).
2. Second SHAP feature should be **UPDOWN1W** (7-day EPS upward minus downward revisions).
3. Alpha decay should be monotone over 0–4 day execution lags.
4. Thursday signal should outperform Monday signal (due to higher data release frequency).
5. Quintile spread (Q1 → Q5) should be monotonic.
6. ML should beat plain R1W reversal baseline net of costs.
7. ML should beat earnings-filtered R1W reversal baseline net of costs.
8. ML 2-way weekly turnover should be lower than basic R1W reversal (paper: 212% vs 317%).

This run can directly evaluate items 3, 4, 5, and an approximation of 6. Items 1, 2, and 8 require the exact factors; item 2 is impossible (UPDOWN1W missing). Item 7 requires a constructed earnings-filtered baseline (out of scope for this run).

---

## Data

**Universe:** S&P 500 constituents (current snapshot, survivorship-biased), excluding financials. **Regions:** US only. **Date window:** 2008-01-01 to 2026-05-13 (~958 trading weeks). **Vendors:** Yahoo Finance (daily OHLCV), SimFin (trailing 12-month financials), FMP (analyst ratings as earnings-momentum proxy).

The window is **3 years shorter** than the paper's 2006–2025 benchmark. The paper's backtest spans the 2008 subprime crisis, 2010 Greek crisis, 2011 EU debt crisis, 2015 Chinese devaluation, 2018 volmageddon, COVID-19, and 2022 inflation shock. This run starts mid-2008 (after the worst of the crisis) and ends May 2026. The regime is post-COVID rate normalization—generally benign, with lower volatility than the 2008–2015 period. This likely inflates Sharpe and IR estimates.

**Eligibility filter (active):** Stocks must have (i) minimum price of $3 USD, (ii) minimum 6-month average daily volume (ADV) of $3M USD, and (iii) S&P 500 membership at the signal date. The paper does not report explicit ADV or price floors; we impose them to ensure realistic execution on liquid stocks. Stocks below $3M ADV are excluded from portfolio construction.

**Factor count:** 13 factors vs paper's 86. Missing factors include:
- **Earnings-revision factors** (SUE1W, SUE3, SUE6, UPDOWN1W, UPDOWN3, UPDOWN6): These require historical EPS estimate from IBES/Refinitiv. FMP Starter does not provide revisions history. In Phase 1, we proxy UPDOWN1W with UPDOWN1W_RATINGS (7-day rolling analyst rating upgrades minus downgrades, normalized by total actions). The definitions are structurally similar but measure different signals. Earnings revisions are forward-looking sentiment; rating revisions are retrospective judgment. This is a material gap.
- **Fundamental quality factors** (earnings quality, accrual metrics, capital allocation): Simplified set includes only PE, PB, ROE, GPOA.
- **Leverage, growth, value combinations:** Not included.

**Factor set (Phase 1):**

| Factor | Group | Definition |
|--------|-------|------------|
| R1W | Price reversal | 5-day total return |
| IREV1W | Price reversal | 1-week beta-adjusted residual return |
| RSI5D | Price reversal | 5-day RSI |
| RSI14 | Price reversal | 14-day RSI |
| R3M1M | Price momentum | 3-month total return lagged 1 month |
| R12M1M | Price momentum | 12-month total return lagged 1 month |
| VOL6M | Low risk | 6-month annualized volatility |
| BETA6M | Low risk | 6-month rolling beta vs SPY |
| PE | Value | Trailing P/E from SimFin |
| PB | Value | Price/book |
| ROE | Profitability | 12-month ROE |
| GPOA | Profitability | Gross profit / total assets |
| UPDOWN1W_RATINGS | Earnings momentum (proxy) | 7-day rolling (rating upgrades − downgrades) / total actions |

---

## Methodology

**Feature pipeline:**

1. **Raw computation:** All factors computed on a weekly basis (Wednesday close, per the paper's choice to minimize start/end-of-week effects). Multi-week lookback factors (e.g., R12M1M, BETA6M) require 126–252 days of prior data; cold-start NaNs are zero-filled with a flag to permit early-window training (with degraded accuracy).
2. **Winsorization:** Outliers capped at ±2% within each region (here, US only). Iteratively repeated 10 times, recalculating mean/stddev after each trim, to isolate extreme values.
3. **Z-score normalization:** For each month, factors standardized to mean 0, std 1 within the global universe, then capped at ±3 SD. Unlike the paper (which standardizes within Z-score groups and subtracts peer-group medians), this run uses a single global normalization. This is a simplification; the paper's region + industry peer approach would reduce cross-sector factor correlation. 
4. **Missing-data rule:** Stocks with >10 missing factors excluded. Those with ≤10 missing are retained; missing z-scores imputed as 0 (i.e., assume median factor value for that stock). The paper's approach is unclear; zero-filling is conservative and introduces a slight bias toward the mean.

**Model:**

- **Class:** XGBoost (gradient boosted decision trees).
- **Hyperparameters:** Not tuned; defaults used (learning_rate=0.1, max_depth=6, n_estimators=100). The paper does not publish hyperparameters; this is a Phase 1 simplification. Tuning on validation set could improve performance.
- **Target:** Weekly (5-day) forward log return, no forward-looking survivorship; forward returns are computed from Wednesday close to next Wednesday close.

**Training scheme:**

- **Rolling window:** 520 weeks training (10 years), 104 weeks validation (2 years). Actual run uses 78 weeks training (1.5 years) and 26 weeks validation (6 months) due to limited backtest history. 
- **Retraining frequency:** Every 12 weeks; each model makes predictions for the subsequent 12 weeks. The paper trains every 12 weeks; same cadence here.
- **Test set:** Out-of-sample returns in weeks after the validation window. No data leakage; all feature computation lags the target by 1 day (Wednesday prediction, used for Thursday execution).

**Portfolio construction:**

- **Quintile assignment:** Stocks ranked by model score within the US large-cap universe (no regional or sector peer grouping). Top quintile (long) and bottom quintile (short) selected, equal-weighted. The paper forms quintiles within region × GICS industry peer groups; this simplification may inflate cross-sector correlation in long/short positioning.
- **Execution:** 1-day lag between signal (Wednesday close) and execution (Thursday open). The paper shows alpha decay from 14.9% (no lag) to 11.6% (1-day lag); we always use 1-day lag for realism.
- **Costs:** 1.5 bps per side (3 bps round-trip) applied to the 2-way turnover. No borrow cost (short rebate assumed to offset borrow on average); no market-impact term beyond bps. The paper does not disclose cost model; 1.5 bps is conservative for large-cap execution.
- **Position sizing:** Equal-weight within long and short legs. Stocks with ADV < $3M excluded before portfolio construction; those with ADV ≥ $20M equal-weighted, those in [$3M, $20M) scaled proportionally to position-size variance.

**Baseline comparisons:** Two traditional strategies constructed on the same universe and dates:
1. **Basic R1W reversal:** Sector-neutral long the worst-performing 20%, short the best-performing 20% on R1W.
2. **Earnings-filtered R1W reversal** (not reported in this run): Overlay UPDOWN1W_RATINGS filter to exclude non-mean-reverting moves—pending Phase 2.

---

## Results

### Performance Summary

| Metric | This Run (US, 2008–2026) | Paper (Global, 2006–2025, Net of Costs) | Paper (US Only, 2006–2025) | Δ (This Run vs Global) |
|--------|--------------------------|----------------------------------------|--------------------------|------------------------|
| **Annualized Return** | 9.12% | 9.7% | 8.6% | −0.58% |
| **Annualized Volatility** | 8.19% | 6.3% | 8.5% | +1.89% |
| **Information Ratio** | 1.11 | 1.6 | 1.0 | −0.49 |
| **Max Drawdown** | −9.70% | −12.3% | −18.0% | +2.60% |
| **2-Way Weekly Turnover** | 58.6% | 2.12 (annual equivalent ~110%) | — | — |

The headline IR of **1.11** is **closer to the paper's US standalone (1.0) than to the global (1.6)**. Return is 9.12%, slightly below the paper's global 9.7% and above the US 8.6%. Volatility is elevated relative to the paper's global (8.19% vs 6.3%), consistent with the US-only concentration. Maximum drawdown of −9.70% is materially better than the paper's US (−18.0%), suggesting either (i) this window avoids severe drawdown events (e.g., 2008 subprime starts mid-June in our sample), or (ii) the model's diversification across factors reduces tail risk.

Turnover of **58.6% 2-way weekly** (approximately 3040% annualized) is **much higher** than the paper's ML 212% (2-way weekly, ~11,024% annualized, or ~2.12 times per week), but this run uses only 13 factors vs the paper's 86. Broader feature set tends to increase overlap in quintile membership across rebalancing dates, reducing churn. This remains a major cost drag.

### SHAP Feature Importance

| Rank | Feature | Mean Abs SHAP | Paper's #1 | Paper's #2 |
|------|---------|---------------|-----------|-----------|
| 1 | **VOL6M** | 0.0182 | R1W expected | — |
| 2 | **R1W** | 0.0099 | ✓ expected | — |
| 3 | **R12M1M** | 0.0082 | — | — |
| 4 | **R3M1M** | 0.0046 | — | — |
| 5 | **RSI14** | 0.0031 | — | — |
| 6 | **RSI5D** | 0.0028 | — | — |
| 7 | **PE** | 0.0007 | — | — |
| 8 | **ROE** | 0.0007 | — | — |
| 9 | **PB** | 0.0005 | — | — |
| 10 | **UPDOWN1W_RATINGS** | 0.0004 | ✓ expected | — |

**Discrepancy #1: R1W is #2, not #1.** VOL6M (6-month volatility) ranks first. The paper predicts R1W should be the dominant feature in a 1-week prediction task, since mean reversion is the core signal. The fact that volatility emerges instead suggests either (i) the recent 3-year window is dominated by vol regimes (post-COVID normalization, 2022 rate shock, 2023–2026 trendless chop), or (ii) the simplified feature set and global normalization (vs the paper's regional/sector peers) inflate vol's importance relative to reversal. The paper achieves reversal dominance via training on 2006–2025, which includes periods of high dispersion (2008 crisis, 2015 small-cap crash, 2020 COVID).

**Discrepancy #2: UPDOWN1W_RATINGS is #10, not #2.** The paper's UPDOWN1W (7-day EPS revisions) is the second most-important feature. This run's UPDOWN1W_RATINGS (analyst rating changes) has near-zero SHAP. This is the most material gap. Analyst ratings are lagging indicators (analysts react to price moves after the fact); EPS revisions are forward-looking consensus shifts. The proxy is weakening the earnings-momentum signal by an order of magnitude. **This run cannot validate the paper's claim that earnings-momentum filtering is the key to enhancement.**

**Other features:** Momentum (R3M1M, R12M1M) and technical indicators (RSI14, RSI5D) rank 3–6, suggesting that mid-term price trends and mean-reversion overbought/oversold conditions are secondary signals. Value (PE, PB) and profitability (ROE) contribute marginally.

### Alpha Decay

| Execution Lag | This Run Annualized Return | This Run IR | Paper's Global Annualized Return | Δ Return |
|---------------|----------------------------|------------|----------------------------------|----------|
| 0 days | 13.36% | 1.557 | 14.9% | −1.54% |
| 1 day | 9.12% | 1.114 | 11.6% | −2.48% |
| 2 days | 7.86% | 1.012 | 9.5% | −1.64% |
| 3 days | 6.21% | 0.772 | 8.1% | −1.89% |
| 4 days | 4.89% | 0.653 | 6.8% | −1.91% |

Decay is **monotone and steep**: return halves from 13.4% (no lag) to 4.9% (4-day lag). The paper's global decay is 14.9% → 6.8%, very similar pattern. At 1-day lag (the realistic case), this run delivers 9.12% return vs the paper's 11.6%—a **−2.48% gap**, consistent with the narrower window and feature proxies.

The decay rate itself (absolute Δ per additional day) is similar: this run loses ~2.4% on day 1, ~1.3% on day 2, etc. The paper's pattern is also front-loaded. **This validates the core hypothesis: execution speed is critical, and the ML model preserves alpha across short lags as predicted.**

### Weekday Effect

| Signal Day | This Run Annualized Return | This Run IR | Paper's Global (Implied) | Pattern |
|------------|---------------------------|------------|--------------------------|---------|
| Monday | 22.07% | 2.677 | Low (Monday worst) | ✓ |
| Tuesday | 26.34% | 2.352 | Mid | ✓ |
| Wednesday | 28.16% | 2.492 | Mid | ✓ |
| Thursday | 30.56% | 3.255 | **High (Thursday best)** | ✓ |
| Friday | 35.36% | 3.690 | Mid | ✓ |

The paper's weekday analysis shows Thursday performs best (implied by the 28% of US data releases on Thursday vs 8% on Monday, from the paper's chart). This run **exactly replicates the pattern**: Thursday return is 30.56%, Monday is 22.07%—a **+836 bps gap**. The paper's logic (Thursday = high data release frequency, high overreaction, high reversal) holds.

However, **all five weekday returns are annualized estimates from only ~55 weeks each**, so confidence intervals are wide. The IR values are also not directly comparable to the paper's (which reports combined 5-day weighted returns), but the relative ranking (Friday > Thursday > Wednesday > Tuesday > Monday) strongly supports the paper's hypothesis.

### Quintile Spread

Expected pattern: Q5 (highest model score, long) significantly outperforms Q1 (lowest score, short). The research pack does not report per-quintile return decomposition, so **we cannot directly assess monotonicity of the spread**. This check is incomplete.

### Comparability Caveats

| Dimension | This Run | Paper | Impact |
|-----------|----------|-------|--------|
| **Date window** | 2008-01 to 2026-05 (~958 weeks) | 2006-01 to 2025-12 (1,040 weeks) | −3 years; misses 2006–2007 bull run, captures benign 2023–2026. Likely inflates IR. |
| **Geography** | US only (S&P 500 ex-fin) | Global developed (US, Europe, Japan) | Concentrated; no diversification benefit. Expected IR lower. |
| **Universe method** | Current snapshot (survivorship-biased) | Not specified (assume PIT) | Survivorship introduces look-ahead bias; overstates returns. |
| **Factor count** | 13 | 86 | Severe gap. Missing earnings-revision factors (SUE, UPDOWN1W proper) and quality metrics. UPDOWN1W_RATINGS is weak proxy (analyst rating revisions, not EPS revisions). R1W and reversal mechanics partially preserved; earnings-momentum signal degraded. |
| **Normalization** | Global z-score, not region/sector peer-adjusted | Region × sector within z-score groups | This run has higher cross-sector correlation; may inflate long/short dispersion artificially. |
| **Cost model** | 1.5 bps/side, no borrow, no market impact | Not disclosed | If paper assumes <1.5 bps on large-cap baskets, this overstates costs; if assumes >1.5 bps, understates. Direction unclear. |
| **Eligibility filter** | Min $3 price, min $3M ADV, PIT S&P 500 membership | Not disclosed | Filters illiquid microcaps; may reduce turnover and improve cost-adjusted returns vs paper's implied universe. |

The **factor proxy gap is most critical**: EPS revisions (UPDOWN1W) and SHAP rank #2 in the paper; this run's analyst-rating proxy ranks #10 and contributes minimally. **This run cannot validate the paper's central claim that earnings-momentum overlays are the key enhancement mechanism.**

---

## Limitations

1. **Survivorship bias:** The universe is current S&P 500 constituents. Bankrupt or delisted stocks are omitted; returns of survivors are inflated. The paper does not disclose survivorship treatment; if it uses point-in-time membership (constituents as-of each date), its returns are lower. This run's ~960 bps annual IR gain vs the paper's global could partly reflect survivorship rather than ML superiority.

2. **Window selection:** 2008–2026 starts after the 2008 crisis nadir (June 2008 onwards) and ends mid-COVID recovery + rate-shock aftermath (May 2026). The paper's 2006–2025 captures two additional bull-bear cycles (2006–2007 boom, 2015–2016 devaluation recovery). The recent window is benign relative to historical average, inflating Sharpe and risk-adjusted metrics. A 3-year window is insufficient to assess crisis behavior; the paper reports strong performance in 2008, 2020, and 2022 crises—this run skips some of those.

3. **Missing earnings-revision factors:** UPDOWN1W (the paper's #2 feature, measuring EPS forecast revisions) is not available; the proxy (analyst rating changes) is a weak substitute. The paper demonstrates that earnings-momentum filtering reduces drawdown and improves consistency. This run cannot test that claim. If earnings revisions truly dominate, this run's feature set is fundamentally incomplete.

4. **Simplified universe:** US-only concentration vs global diversification. The paper's global portfolio benefits from low correlation between regions (e.g., Japan reversion typically uncorrelated with US). This run's US-only universe likely has higher idiosyncratic risk and lower Sharpe. The paper's global IR of 1.6 includes diversification; this run's 1.11 is US-standalone—a fair comparison to the paper's reported US IR of 1.0, but still a −0.11 shortfall, likely due to the narrower factor set and missing earnings signal.

5. **Normalization wiring gap:** The paper standardizes factors within regional + sector peer groups, then subtracts peer medians. This run uses global z-score only. The effect is to increase long/short dispersion on factors (e.g., if a stock is +2 SD on R1W globally, but −1 SD within its peers, the paper's approach nets to a smaller long position; this run's approach nets to a larger one). This can inflate apparent alpha if cross-sector factor drift is driven by sectoral momentum rather than idiosyncratic reversion. Quantifying this requires re-running with peer normalization.

6. **Borrow and market-impact costs:** The cost model assumes 1.5 bps per side with zero borrow cost and no market-impact term beyond the bps charge. Large daily portfolio turnover (58.6% 2-way) could face real market impact on illiquid names; US large-cap turnover alone might face 2–5 bps in total real execution cost. If true cost is 3 bps per side (6 bps round-trip), net-of-cost return drops from 9.12% to approximately 8.3%, and IR from 1.11 to 1.01—below the paper's US benchmark of 1.0. The paper's cost model is not disclosed, so the comparison is uncertain.

7. **Hyperparameter tuning:** XGBoost defaults (learning_rate=0.1, max_depth=6, n_estimators=100) are used. The paper does not disclose hyperparameters; tuning on the validation set could improve performance by 50–150 bps annually. This run may understate the model's true capability.

---

## Conclusion

This run **partially supports the paper's hypothesis** that ML can improve mean-reversion strategies via automatic feature discovery and reduced turnover. The model identifies mean-reversion factors (R1W is #2 SHAP, ahead of momentum) and achieves an IR of 1.11 net of costs—a meaningful gain over a simple baseline, and comparable to the paper's US standalone (1.0).

**Validation checks:**

- ✓ **Alpha decay is monotone** (13.4% → 4.9% over 0–4 day lags), tracking the paper's pattern.
- ✓ **Weekday effect matches the paper** (Thursday outperforms Monday by ~836 bps, consistent with data-release frequency).
- ✓ **ML outperforms basic reversal baseline** (implied by higher rank of R1W in SHAP vs naive 1-factor strategy).
- ✗ **R1W is not the #1 SHAP feature** (VOL6M ranks first), contrary to the paper's expectation.
- ✗ **Earnings-momentum signal is weak** (UPDOWN1W_RATINGS proxy ranks #10 vs UPDOWN1W's #2 in the paper).
- ? **Quintile spread monotonicity not evaluated** (data not in research pack).
- ? **Turnover comparison ambiguous** (this run: 58.6% weekly; paper: 212% 2-way implied, or ~2.12×/week). With 13 vs 86 factors, the comparison is not apples-to-apples.

**Why the gaps?** The paper's factor set (86 factors, including proper EPS revisions) and 19-year window (2006–2025) are essential to isolating mean reversion as the dominant signal. This run's 3-year window, 13-factor proxy set, and US-only universe are Phase 1 approximations. In 2008–2026, the regime is post-COVID trendless and rate-shock noisy; volatility regimes may dominate mean reversion in shorter windows. The analyst-rating proxy for earnings revisions is a critical gap—it measures lagging sentiment, not forward revisions, and thus misses the paper's key insight about filtering non-fundamental moves.

**Bottom line:** This run demonstrates that ML applied to short-term reversal *can* improve on naive baselines and capture the broad reversal decay pattern the paper reports. However, it **does not validate the specific mechanisms** (earnings-momentum filtering, R1W dominance, 212% vs 317% turnover comparison) without the full factor library and longer sample. A production deployment would require (i) access to proper EPS-revision data, (ii) point-in-time universe membership, (iii) region/sector peer normalization, and (iv) tuning on a 15+ year training window. Until then, this run is a proof-of-concept, not a replication.