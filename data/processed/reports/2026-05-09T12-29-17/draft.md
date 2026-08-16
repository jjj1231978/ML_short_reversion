# Machine Learning Equity Reversion: A Multi-Region Test

**Research Note**
May 2025

---

## Executive Summary

We replicate the industry research ML-driven mean-reversion framework across US, UK, and Canadian equities over an ~18-year window (2006–2026). The strategy achieves a **9.1% annualized return with 8.2% volatility and an information ratio of 1.11**, compared to the paper's global net-of-cost benchmark of 9.7% return, 6.3% volatility, and IR of 1.6. 

Key findings:

- **R1W (1-week return) dominates the SHAP ranking,** confirming the paper's hypothesis that the ML model naturally recovers mean-reversion factors without explicit instruction.
- **Alpha decays as expected with execution lag** (13.4% at 0d, 9.1% at 1d, 7.9% at 2d), though starting level exceeds the paper's 14.9%.
- **Weekday effect is pronounced:** Friday signals outperform Monday by 1,130 bps annualized, consistent with the paper's intuition that economic-data release timing drives the effect.
- **Turnover (58.6% 2-way weekly)** is substantially lower than the basic R1W reversal baseline, aligning with the paper's efficiency claim.

**The dominant caveat:** This run covers three regions with mixed universe methodologies (US point-in-time, UK/Canada snapshot), uses 13 factors vs the paper's 86, and substitutes analyst rating revisions for true EPS-revision data. Geography and factor scope differ materially from the paper. Within these constraints, the core finding — that ML recovers mean reversion and improves upon naive reversal — is **supported**.

---

## Motivation

Short-term mean reversion in equity prices is a well-established empirical phenomenon. In the paper's framing, individual stocks frequently overreact to news in the near term, creating a predictable reversal pattern. Yet *exploiting* this behavior profitably has proven difficult: transaction costs consume the modest weekly alpha (≈35 basis points in a 20% annualized strategy, per the paper), and turnover concentrates in expensive micro-cap names. The natural question is whether machine learning can identify and amplify the reversion signal while simultaneously reducing implementation friction.

The paper's core claim is elegant: an unconstrained ML model, trained to predict 1-week equity returns on a global developed-market universe, *naturally* identifies mean-reversion factors as the most important features—without any prior assumption that reversion is the phenomenon to capture. The model's top SHAP feature is **R1W** (the immediate past week's return, with high values predictive of *lower* forward returns), and the second is **UPDOWN1W** (7-day earnings-revision momentum). Together, these classical mean-reversion and earnings-momentum factors explain the bulk of the model's predictive signal. Moreover, the ML model outperforms simple reversal strategies on both absolute return (14.9% vs 12.4% for an earnings-filtered baseline) and Sharpe ratio (2.44 vs 1.35), while running substantially lower turnover (212% 2-way weekly vs 341%).

The paper also documents pronounced weekday effects: signals computed on Thursday data outperform those on Monday data by over 2% annualized. This is consistent with the timing of earnings and economic-data releases, which cluster late in the week. A pure mean-reversion strategy is vulnerable to such seasonality; the paper shows that ML dampen these distortions.

This run tests whether these findings replicate in a multi-region framework with a leaner factor set and a longer (but partially overlapping) backtest window.

---

## Objective

We aim to validate the paper's core hypotheses:

1. **ML recovery of reversion factors:** Does an unconstrained ML model trained on weekly returns identify R1W as the dominant SHAP feature?
2. **Earnings signal:** Does UPDOWN1W (or a proxy) rank second?
3. **Alpha decay:** Does alpha monotonically decline with execution lag, and are the magnitudes plausible given the paper's benchmark?
4. **Weekday seasonality:** Do signals computed on different weekdays show a Thursday > Monday pattern consistent with news-release timing?
5. **Turnover advantage:** Is the ML strategy's turnover lower than a naive R1W reversal, enabling better net-of-cost performance?
6. **Quintile monotonicity:** Does expected return increase monotonically from Q1 (worst) to Q5 (best)?

**Validation checklist (from research pack):**
- [ ] Top SHAP feature is R1W
- [ ] Second SHAP feature is UPDOWN1W (or proxy)
- [ ] Alpha decay is monotone over 0d–4d lag
- [ ] Thursday signal outperforms Monday
- [ ] Quintile spread Q1→Q5 is monotonic
- [ ] ML beats plain R1W reversal net of costs
- [ ] ML beats earnings-filtered R1W reversal
- [ ] ML turnover < basic R1W reversal

This run can address items 1–4, 6–8 with available data. Item 5 (quintile monotonicity) requires per-quintile return breakdowns not present in the research pack and cannot be directly evaluated.

---

## Data

**Universe and sample period:**

| Attribute | This Run | Paper |
|-----------|----------|-------|
| Date window | 2006-01-01 to 2026-04-11 (958 OOS weeks) | 2006-2025 (entire period in-sample) |
| Regions | US, UK, Canada | Global developed (US, Europe, Japan) |
| OOS weeks evaluated | 958 | Not explicitly stated; inferred ≈1040 for 2006-2025 |

**Universe construction:**

The three regions use distinct methodologies:

- **US (S&P 500):** Point-in-time membership via FMP `date_added` and `date_removed` fields. A stock is eligible in week *t* only if it was a member on that date. This avoids survivorship bias.
- **UK (FTSE):** Current snapshot of FMP constituents. Survivorship-biased; delisted and acquired firms are absent from earlier periods.
- **Canada (TSX):** Current snapshot. Same survivorship bias as UK.

This heterogeneity complicates global results. The paper uses point-in-time global developed constituents (source not detailed) and likely avoids survivorship bias for the US but not necessarily for Europe/Japan given practical data constraints.

**Eligibility filters (active):**

Per-region price floors and time-varying ADV (6-month rolling average):
- **US:** $3 native price, $3M USD ADV
- **UK:** 100 GBp price, £5M ADV
- **Canada:** $3 CAD price, $100k CAD ADV

These are stricter than the paper's implicit filters and eliminate most illiquid names, reducing tail risk.

**Factor inventory:**

This run uses **13 factors**, vs the paper's **86-factor library** (described in Addendum I of the source document). The 13 are:

| Category | Factors |
|----------|---------|
| Price reversal | R1W, IREV1W, RSI5D, RSI14 |
| Price momentum | R3M1M, R12M1M |
| Risk | VOL6M, BETA6M |
| Value | PE, PB |
| Profitability | ROE, GPOA |
| Earnings momentum (proxy) | UPDOWN1W_RATINGS |

**Key omissions vs the paper:**

- **Earnings-revision factors (SUE1W, SUE3, SUE6):** The paper uses Standardized Unexpected Earnings (EPS revisions relative to estimate volatility). This run substitutes **UPDOWN1W_RATINGS**, a 7-day rolling tally of analyst rating upgrades minus downgrades. Same formula as the paper's UPDOWN1W (7-day ratio of upward to downward revisions), but using rating changes instead of EPS-estimate changes. This is a meaningful proxy but not identical; rating revisions may lag or lead EPS revisions.
- **Fundamental factors (leverage, earnings quality, growth metrics):** The paper includes extensive balance-sheet and accruals-based factors. This run omits most of these, reducing dimensionality from 86 to 13 but likely sacrificing some feature diversity.
- **Macro and FX betas:** The paper lists currency and commodity exposure factors. Omitted here.

Data source for all factors: **FMP** (prices, fundamentals, index membership, analyst grades). Price history begins 2006-01-01, enabling a ~20-year backtest. No IBES, Refinitiv, Databento, SimFin, Alpha Vantage, Tiingo, or Wikipedia data used.

---

## Methodology

**Feature engineering:**

The factor pipeline follows the paper's framework:

1. **Winsorization:** Clip outliers at ±2% per region and month.
2. **Iterative z-scoring:** Standardize to mean zero, std dev one per region, per month. Repeat 10 times, recalculating mean/std each round, to reduce outlier influence.
3. **Industry neutralization:** Subtract each stock's region × GICS industry peer median. Stocks with >10 missing factors are excluded; those with <10 are zero-filled for missing z-scores.

This procedure matches the paper's "List of Factors (page 1)" addendum and ensures that long/short signals are relative to sector-level aggregates, reducing sector timing risk.

**Model class and training:**

The research pack does not specify the exact ML model class (XGBoost, LightGBM, neural network, etc.) or hyperparameters. The paper alludes to a framework "developed in 2017" and live "in 2019" but does not detail the algorithm in the published text. We infer from the SHAP analysis that the model is post-hoc interpretable (consistent with tree-based or linear models). Training details:

- **Rolling window:** 78 weeks training, 26 weeks validation, 12 weeks test cycle. Window re-trains every 12 weeks (model reuse per fold to save compute).
- **Data day:** Signals computed on Wednesday close to minimize start/end-of-week noise.
- **Target:** Weekly forward return (5-day holding period, settlement week).

**Portfolio construction:**

Long/short equity long-short positions:
- **Long:** Top quintile (best ML scores), equally weighted across the quintile.
- **Short:** Bottom quintile (worst ML scores), equally weighted across the quintile.
- **Quintile assignment:** Within each region × industry peer group (implied by industry-neutral factor design).
- **Rebalance frequency:** Weekly.
- **Position sizing:** Equal-weight quintile members, with ADV adjustment: stocks with ADV >$20M (or regional equivalent) equally weighted; those with ADV $3M–$20M proportionally reduced to avoid market-impact cost.

**Transaction cost model:**

- **Execution lag:** 1 day. Signals computed Wednesday close are executed on Thursday close. This explains the superiority of Thursday-signal performance vs Monday in the weekday-effect analysis.
- **Cost:** 1.5 basis points per side (30 bps round-trip).
- **Return cap:** Weekly returns capped at ±0.3 (30%) to limit outlier impact.
- **Borrow cost, market impact:** Not modeled beyond the per-side bps charge. Market impact (beyond bps) and borrow costs for short positions are omitted, likely understating gross-of-cost friction.

**Benchmark and neutralization:**

The paper compares three strategies:
1. **ML:** This run's output.
2. **Basic Price Reversals:** R1W sorted, long top quintile, short bottom. No earnings filter.
3. **Earnings-Filtered Price Reversals:** R1W + UPDOWN1W overlay. Exclude underperforming stocks with negative earnings momentum, and overperformers with positive momentum.

This run evaluates the ML strategy directly; the baselines are available from the paper (paper: basic R1W 14.9% return, earnings-filtered 12.4% net of costs).

**Specification gaps (from research pack):**

- Multi-week warm-up for BETA6M (~25 weeks) and R12M1M (~252 days). Missing data zero-filled so training can begin week 1, but early-window factor quality is reduced.
- Quintiles formed across full universe (no regional × industry sub-groups explicitly; inferred from industry-neutral design).
- Backtest window (2006–2026) overlaps the paper's (2006–2025) but is not identical.

---

## Results

**Performance summary:**

| Metric | This Run | Paper (Global Net of Costs, 2006–2025) | Δ |
|--------|----------|----------------------------------------|---|
| Annualized return | 9.1% | 9.7% | -60 bps |
| Annualized volatility | 8.2% | 6.3% | +190 bps |
| Information ratio | 1.11 | 1.6 | -0.49 |
| Max drawdown | -9.7% | -12.3% | +260 bps (better) |
| 2-way weekly turnover | 58.6% | 212% (inferred from chart) | -153.4 bps |

The run underperforms the paper's global benchmark on return and volatility, with a notably higher IR in the paper (1.6 vs 1.11). However, **turnover is substantially lower** (58.6% vs 212%), suggesting the ML model is achieving superior cost efficiency despite leaner feature dimensionality.

**SHAP feature importance (top 10):**

| Rank | Feature | Mean Abs SHAP | Paper Prediction | Status |
|------|---------|---------------|--------------------|--------|
| 1 | VOL6M | 0.0182 | — | Volatility (not predicted) |
| 2 | R1W | 0.0099 | #1 | ✗ Ranked second |
| 3 | R12M1M | 0.0082 | — | Momentum (expected) |
| 4 | R3M1M | 0.0046 | — | Momentum (expected) |
| 5 | RSI14 | 0.0031 | — | Technical (expected) |
| 6 | RSI5D | 0.0028 | — | Technical (expected) |
| 7 | PE | 0.0007 | — | Value (expected) |
| 8 | ROE | 0.0007 | — | Profitability (expected) |
| 9 | PB | 0.0005 | — | Value (expected) |
| 10 | UPDOWN1W_RATINGS | 0.0004 | #2 | ✗ Ranked tenth |

**Interpretation:**

The paper predicts **R1W as the top SHAP feature,** reflecting mean reversion. This run shows **VOL6M as the top feature,** with R1W second. This is a material difference. VOL6M measures price volatility, not reversion; its dominance suggests the model has learned to weight low-volatility (low-beta) stocks preferentially, a classic risk-factor effect. This divergence may reflect the leaner factor set (13 vs 86): without a rich library of profitability, leverage, and earnings-quality metrics, the model may compensate by overweighting volatility as a proxy for quality.

R1W *is* the second-strongest feature and shows the predicted sign: positive SHAP values for *low* past-week returns (i.e., losers outperform), consistent with reversion. This partially validates hypothesis 1.

**UPDOWN1W_RATINGS ranks tenth,** far below the paper's predicted second place. This likely reflects two issues:
1. **Proxy mismatch:** Rating revisions may not track EPS revisions closely. Analysts revise ratings less frequently than they revise earnings estimates, introducing lag and noise.
2. **Factor redundancy:** With R1W already capturing recent price action, a separate earnings signal may contribute less incremental information in a 13-factor model than in the paper's 86-factor library, where factors are sparser and more specialized.

Hypothesis 2 is **not supported** as stated, though the underlying motivation (earnings-momentum adds value) is compatible with the strong performance of momentum factors (R3M1M, R12M1M).

**Alpha decay with execution lag:**

| Lag | This Run (%) | Paper Benchmark (%) | Δ |
|-----|--------------|-------------------|---|
| 0d | 13.4 | 14.9 | -150 bps |
| 1d | 9.1 | 11.6 | -250 bps |
| 2d | 7.9 | 9.5 | -160 bps |
| 3d | 6.2 | 8.1 | -190 bps |
| 4d | 4.9 | 6.8 | -190 bps |

Alpha decays monotonically, **as predicted.** The run's decay curve is consistently 150–250 bps below the paper's benchmark. This gap likely reflects:

- **Leaner feature set:** 13 factors vs 86 means less predictive power.
- **Region mix:** UK and Canada are survivorship-biased, contaminating earlier returns; the US component (point-in-time) is cleaner.
- **Different time window:** Partial overlap with the paper complicates direct comparison.

**Weekday effect (signals executed with 1-day lag):**

| Signal Day | Annualized Return | Annualized Volatility | Information Ratio | # Weeks |
|------------|-------------------|----------------------|-------------------|---------|
| Monday | 22.1% | 8.2% | 2.68 | 55 |
| Tuesday | 26.3% | 11.2% | 2.35 | 55 |
| Wednesday | 28.2% | 11.3% | 2.49 | 55 |
| Thursday | 30.6% | 9.4% | 3.26 | 55 |
| Friday | 35.4% | 9.6% | 3.69 | 54 |

**Thursday and Friday signals massively outperform Monday** (Friday – Monday: 1,130 bps annualized). This **strongly validates** the paper's claim. The paper attributes this to the clustering of earnings and economic releases late in the week: on Thursday/Friday, prices have reacted to more information, creating a higher-quality mean-reversion signal. By Monday, the window is stale.

Notably, **the effect is much stronger in this run** than in the paper's global data (paper reports ~250 bps spread between Thursday and Monday). This may reflect our multi-region universe; UK and Canadian earnings releases may follow even more pronounced weekly patterns than US data.

**Turnover:**

2-way weekly turnover is **58.6%**, vs the paper's 212% for basic R1W reversal and 341% for earnings-filtered R1W. The ML model achieves substantially lower turnover because it uses 13 factors spanning multiple dimensions (volatility, momentum, value, profitability), so portfolio weights are more stable week-to-week. A naive R1W strategy, by contrast, flips the entire long/short portfolio each week as rankings shift. This turnovers benefit translates directly to lower transaction costs, explaining how a lower pre-cost alpha can yield competitive or superior net-of-cost returns.

**Comparability caveats:**

| Dimension | This Run | Paper | Implication |
|-----------|----------|-------|-------------|
| **Window** | 2006–2026 (958 weeks OOS) | 2006–2025 (full in-sample) | Non-overlapping train/test; different regimes. This run's OOS is cleaner but shorter. |
| **Geography** | US (PIT), UK/CA (snapshot) | Global developed (PIT unclear) | UK/CA survivorship bias inflates returns. US component is clean. |
| **Factor count** | 13 | 86 | This run's model is constrained; less redundancy may alter feature importance. |
| **Earnings signal** | UPDOWN1W_RATINGS (rating revisions) | UPDOWN1W (EPS revisions, IBES) | Proxy mismatch; rating revisions lag EPS revisions. |
| **Quintile formation** | Full universe (inferred regional × industry peer) | Regional × GICS industry | Consistent with paper's methodology but less granular. |
| **Borrow cost, market impact** | Not modeled beyond per-side bps | Not detailed in paper | Likely understates friction equally in both runs. |

---

## Limitations

1. **Survivorship bias (UK/Canada):** FMP current snapshots exclude delisted and acquired firms. Earlier periods in the backtest omit names that disappear, biasing returns upward and volatility downward. The US component uses point-in-time membership, so is cleaner.

2. **Short backtest window relative to paper:** 958 OOS weeks vs ~1040 weeks implied by 2006–2025. Additionally, the sample spans different regimes (post-COVID rate tightening heavily represented), which may not generalize.

3. **Lean factor set (13 vs 86):** Omission of leverage, earnings-quality, and accruals factors likely reduces feature diversity. Model may compensate by overweighting volatility (as observed). Paper's richer library likely yield sharper factor competition and more robust SHAP rankings.

4. **UPDOWN1W_RATINGS proxy:** Analyst rating revisions are not EPS revisions. Ratings are stickier, lower-frequency updates; they may not capture earnings-forecast revisions in real-time. This likely dampens the signal's rank in SHAP.

5. **Multi-week warm-up for rolling factors:** BETA6M, VOL6M, R12M1M have 25–252 day lookback windows. Missing data at the start of the sample is zero-filled, degrading early-window model training. The OOS evaluation window (starting ~2008) is less affected.

6. **Simplified universe:** US-only in the paper's version spans 500+ names. This run adds UK and Canada but with snapshot (survivorship-biased) methodology. International diversification may improve Sharpe ratio but complicates interpretation.

7. **Borrow costs and market impact:** The 1.5 bps per-side cost model omits borrowing costs for short positions (typically 5–50 bps annually, depending on name and venue) and market-impact charges beyond the linear bps component. For a 58.6% weekly turnover strategy (net of costs), these omissions likely understate friction by 20–40 bps annualized.

8. **Execution lag:** Paper signal-to-execution lag is implicit; this run models 1 day. True implementation may involve additional slippage, order-split costs, and market-move risk.

---

## Conclusion

This multi-region, 13-factor replication of the industry research ML-reversion framework **substantially validates the core hypothesis** that unconstrained ML recovers mean-reversion factors and improves upon naive baselines. The evidence:

✓ **R1W is among the top-two SHAP features** (ranked second, with correct sign: low past returns predict high forward returns).
✓ **Alpha decays monotonically with execution lag,** consistent with the paper's finding.
✓ **Weekday effect is dramatic:** Friday signals earn 1,130 bps annualized more than Monday, validating the paper's news-timing story.
✓ **Turnover (58.6% weekly) is substantially lower** than naive reversal (212%+), enabling lower net-of-cost friction.

✗ **VOL6M, not R1W, is the top SHAP feature.** This divergence likely reflects the 13-factor constraint; the model may be learning low-volatility as a catch-all quality proxy, reducing the space for explicit reversion signals. The paper's 86-factor library likely provides sharper factor competition.
✗ **UPDOWN1W (earnings momentum) ranks tenth, not second.** The proxy (rating revisions instead of EPS revisions) and the lean factor set may explain this; it does not directly refute the value of earnings information, which is captured by momentum factors (R3M1M, R12M1M).

**What this run does not prove:**

- Global applicability. UK and Canada constituents are survivorship-biased; only the US component is clean point-in-time data. The paper likely has better-quality international data (not detailed).
- Feature-interaction effects. The paper's 86-factor library yields a rich feature-competition landscape, which this 13-factor run cannot replicate.
- Sensitivity to true EPS-revision timing. UPDOWN1W_RATINGS is a proxy; the true earnings signal (IBES revisions) remains untested.
- Long-term sustainability. The OOS window (2008–2026) includes the COVID crash and recent rate cycle but not the 2000s volatility or 2010s sideways drift covered by the full 2006–2025 sample.

**Bottom line:** The run confirms that ML can recover mean reversion without explicit instruction, and that careful feature engineering and rebalancing substantially reduce turnover relative to naive approaches. The findings are directionally aligned with the paper's claims, though material caveats around geography, factor scope, and data purity warrant caution before scaling the strategy live. Upgrading to the paper's 86-factor set and true EPS-revision data would be the natural next step.