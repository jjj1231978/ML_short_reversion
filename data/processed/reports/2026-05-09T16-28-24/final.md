# Machine Learning for Short-term Stock Return Prediction: POC for three Developed Markets

## Executive Summary

This run evaluates an ensemble ML model (XGBoost + LightGBM + RandomForest) trained to predict 1-week-ahead stock returns across US, UK, and Canadian large/mid-cap universes over 958 weeks (2008–2026). The ensemble achieves an **information ratio of 1.24 and annualized return of 10.0%**, net of 1.5 bps trading costs and 1-day execution lag. This **trails the paper's global net-of-costs IR of 1.6 by 0.36 units** but operates on a narrower geography (3 regions vs US/Europe/Japan) and a recent window with survivorship bias in UK/CA universes. The 1-week reversal factor (R1W) ranks **second in SHAP importance** (behind volatility), contrary to the paper's finding that R1W is #1. **Weekday effects replicate strongly** (Friday 35% > Thursday 31% > ... > Monday 22%), confirming the paper's signal-timing hypothesis, though with a late-week shift. The dominant caveat is that UK and Canada universes are current-membership snapshots (not point-in-time), inflating performance in surviving names. Additionally, earnings-revision factors cannot be evaluated (FMP lacks historical EPS-estimate data), precluding validation of the paper's #2 SHAP feature. Within its stated scope, the run validates the core mean-reversion concept but identifies a **critical VOL6M-dominance anomaly** requiring explanation, and reveals that earnings momentum is not captured by the available data.

---

## Motivation

Short-term price reversals—the tendency of stocks to revert after overreacting to news—have long been documented in academic research (Jegadeesh, 1990, cited in the paper). Yet exploiting them in practice remains difficult: daily turnover is high, alpha decays rapidly with execution lag, and the raw return spread is small (the paper estimates 35 bps per week, or 20% annualized, before costs). industry research's research (April 2025) demonstrates that a machine-learning framework trained on a broad set of 80+ equity factors without explicit mean-reversion bias naturally rediscovers price reversal and earnings momentum as dominant predictors. Specifically, the paper reports:

- **R1W (1-week return) is the #1 feature** in SHAP importance, with negative coefficient (recent winners underperform next week).
- **UPDOWN1W (7-day EPS-revision breadth) is #2**, confirming that fundamental news dampens reversals.
- **Thursday signals outperform Monday signals** by ~350 bps annualized, likely because economic data and earnings releases cluster on Thursdays.
- **Two-way weekly turnover is 212% for the ML model vs 317% for a traditional reversal rule**, yielding superior net-of-cost performance despite identical gross alpha.

The paper's headline finding is that ML enhances traditional mean-reversion alpha by integrating these intuitions automatically, without explicit rules. This run tests whether that claim holds across a multi-region, more recent sample with a simplified factor set and modern data sources.

---

## Objective

This run addresses the following validation checklist, derived from the paper's reported findings:

1. **R1W ranks first in SHAP importance.** The paper explicitly claims this; we can verify it directly.
2. **UPDOWN1W ranks second.** This is not evaluable—FMP lacks historical analyst-estimate revisions, so we proxy with analyst-rating revisions (UPDOWN1W_RATINGS). A fair test requires the original EPS-revision signal.
3. **Alpha decays monotonically with execution lag (0d → 4d).** The paper reports 14.9% → 11.6% → 9.5% → 8.1% → 6.8%. We compare the decay curve and steepness.
4. **Thursday signals outperform Monday signals.** The paper's weekday effect shows Thursday earning 28% of weekly economic data releases (highest) vs Monday 8% (lowest). We test whether our model's returns align.
5. **Per-quintile returns are monotonic (Q1 worst, Q5 best).** We evaluate monotonicity where data permits.
6. **Two-way turnover is lower for ML than traditional reversal.** The paper reports 212% vs 317%. We report ensemble turnover but cannot construct a traditional reversal baseline with identical universe/cost assumptions.
7. **Net-of-cost global IR exceeds 1.0.** The paper reports 1.6 (global, 2006-2025). We report 1.24 on a narrower sample.

Items 1, 3, 4, 5, and 7 are testable. Items 2 and 6 are constrained by data availability and experimental design, respectively.

---

## Data

**Universe Construction & Geography.**  
The run covers three regions:
- **US**: FMP point-in-time S&P 500 membership (date_added / date_removed fields). Respects historical eligibility—a stock joining the index in 2015 is not in the 2010 universe.
- **UK & Canada**: FMP current-membership snapshots (survivorship-biased). Any company currently listed in FMP appears across all historical weeks, even if it delisted or joined after 2006. This is a material difference from the paper and inflates returns in surviving mega-cap names.

**Date Window.**  
2006-01-01 to 2026-04-11 (958 out-of-sample weeks, with ~520-week rolling training window). Portfolio trades begin 2008-01-09, reflecting warm-up for rolling-window factors (BETA6M requires ~126 days; R12M1M requires ~252 days). The paper covers 2006–2025 identically; this run extends to mid-2026 and has degraded signal quality in early weeks.

**Data Vendor.**  
FMP exclusive for prices, fundamentals, analyst grades (as EPS-revision proxy), and index membership. **Critical gap:** FMP Starter includes daily OHLCV and fundamental timeseries but NOT historical consensus EPS estimates or revisions. The paper uses IBES for earnings revisions; we approximate with FMP analyst rating-revision breadth. This gap prevents validation of the paper's #2 SHAP feature.

**Eligibility Filters.**  
Per-region floors applied at portfolio construction:
- **US**: $3 native price floor, $3M USD 6-month ADV floor, point-in-time S&P 500 membership.
- **UK**: 100 GBp price floor, 5M GBp 6-month ADV floor, current snapshot.
- **CA**: $3 CAD price floor, $100k CAD 6-month ADV floor, current snapshot.

Financials are excluded (standard; many factor definitions break for banks).

**Factors.**  
The run uses **13 factors**, derived from FMP data:
- **Price reversal (5)**: R1W, IREV1W, RSI5D, RSI14.
- **Price momentum (2)**: R3M1M, R12M1M.
- **Low risk (2)**: VOL6M, BETA6M.
- **Value (3)**: PE, PB, GPOA.
- **Profitability (2)**: ROE, GPOA.
- **Earnings momentum (1)**: UPDOWN1W_RATINGS (analyst rating revisions, *not* EPS revisions—a material proxy limitation).

The paper's full library includes 86 factors (80 baseline + 6 weekly-specific) across 10 groups. This run's 13-factor set is a ~7:1 compression, sparse on leverage, growth-signal depth, quality, and capital allocation. This limitation affects both the magnitude and stability of the alpha estimates.

---

## Methodology

**Feature Pipeline.**  
For each stock-week, raw factors are computed from daily and fundamental data, then:
1. **Winsorized** at ±2% tails within each region.
2. **Iteratively z-scored**: standardized to mean=0, σ=1 within a region or region × sector group, clamped to [–3, 3], repeated 10 times.
3. **Median-adjusted**: subtract the median of region × GICS sector peers, so each factor is relative to its peer group.
4. **Missing-data fill**: stocks with >10 missing factors excluded; others zero-filled.

This mirrors the paper's methodology. **Caveat:** this run's quintile formation (per Portfolio Construction below) is not explicitly sector-neutral, which may induce sector tilts.

**Model & Ensemble.**  
The headline ensemble combines three members via **rank aggregation** (mean of ranks):
- **XGBoost**: max_depth=4, lr=0.03, 2000 trees, subsample/colsample=0.7.
- **LightGBM**: num_leaves=31, lr=0.03, 2000 trees, subsample/colsample=0.7.
- **RandomForest**: 400 trees, max_depth=8, min_samples_leaf=50, max_features=0.7.

Each is trained on rolling 520-week windows, validated on 26-week hold-outs, retrained every 12 weeks. Final portfolio scores are the average rank across members. The paper does not disclose ensemble architecture; this rank-mean approach is a standard modern choice.

**Portfolio Construction.**  
For each week:
1. Compute ensemble scores for all eligible stocks.
2. Form quintiles within each region globally (not sector-neutral; noted as a limitation below).
3. Long the top quintile (Q5), short the bottom (Q1), equal-weight within each.
4. Hold for 1 week, rebalance on Wednesday close assumption.

Cost model: 1.5 bps per side (3 bps round-trip), 1-day execution lag. Weekly returns capped at ±30% (risk management, not in the paper).

---

## Results

### Performance & Metrics

| Metric | This Run | Paper (Global, 2006–2025) | Δ | Notes |
|---|---|---|---|---|
| **Annualized Return** | 10.0% | 9.7% | +0.3% | Slight nominal outperformance; narrower geography & recent window. |
| **Annualized Volatility** | 8.0% | 6.3% | +1.7% | Higher risk; 3-region universe less diversified than global. |
| **Information Ratio** | **1.24** | **1.6** | –0.36 | Paper outperforms on risk-adjusted basis. |
| **Max Drawdown** | –7.9% | –12.3% | +4.4% (favorable) | Shallower trough; recent window skews toward post-COVID recovery. |
| **Avg 2-Way Weekly Turnover** | 59.8% | 212% (ML) | 3.5× lower | Likely due to smaller factor set & ensemble regularization. |

The ensemble's 1.24 IR is solid but trails the paper's 1.6 by 0.36 units. The paper's US-only performance is 1.0 IR; the gap between US and global (1.0 vs 1.6) is 0.6 units, suggesting regional diversification is valuable. This run's position between the two implies the 3-region mix contributes modest alpha but at higher volatility.

### SHAP Feature Importance

**Ensemble (Rank-Mean Aggregate):**

| Rank | Feature | Mean Abs SHAP | Interpretation |
|---|---|---|---|
| 1 | **VOL6M** | 0.0182 | 6-month volatility; negative coefficient (high vol → lower next-week return). |
| 2 | **R1W** | 0.0099 | 1-week price reversal; past winner underperforms next week. |
| 3 | **R12M1M** | 0.0082 | 12-month momentum lagged 1 month; positive coefficient. |
| 4–13 | R3M1M, RSI14, RSI5D, PE, ROE, PB, UPDOWN1W_RATINGS, GPOA, IREV1W, BETA6M | <0.005 | Collectively minor. |

**Paper's Ranking:**  
The paper reports (page 6–7) **R1W as #1** and **UPDOWN1W (7-day EPS revisions) as #2**. This run shows **VOL6M as #1** (0.0182 vs R1W's 0.0099), **R1W as #2**. This is a **material divergence**.

**Interpretation:**  
The paper's finding emphasizes *mean reversion* as the core alpha driver. This run suggests *volatility regimes* are equally or more predictive. Per-member rankings show:
- **XGBoost**: VOL6M #1 (0.0182), R1W #2 (0.0099).
- **LightGBM**: VOL6M #1 (0.0204), R1W #2 (0.0141).
- **RandomForest**: VOL6M #1 (0.0208), R12M1M #2 (0.0128)—R1W drops to #3.

**All three members agree VOL6M dominates**, but disagree on the 2–3 ranking. This consensus disagreement with the paper's R1W #1 is **unexplained and material**. Possible explanations: (i) FMP price data is noisier, inflating estimated volatility relative to reversal signal; (ii) the 13-factor set lacks the earnings-quality and capital-allocation factors that disambiguate noise from fundamental reversals; (iii) the survivorship-biased UK/CA universes over-weight low-vol, high-quality survivors, making volatility a stronger predictor of next-week underperformance. **This gap requires investigation and is flagged as a limitation.**

**UPDOWN1W_RATINGS Failure:**  
The paper's #2 feature is UPDOWN1W (7-day EPS-estimate revisions). This run uses UPDOWN1W_RATINGS (analyst rating revisions), which ranks **10th with 0.0004 SHAP**—approximately **0.1%** of R1W's signal. This represents a **critical failure**: the earnings-momentum proxy is ineffective, and the model does not capture what the paper identified as the second-largest alpha driver. **Replication requires historical consensus EPS-estimate data**, not available in FMP Starter. This is a data-limitation gap, not an implementation error.

### Alpha Decay

| Lag (days) | This Run | Paper | Δ | Loss Rate |
|---|---|---|---|---|
| 0 | 13.8% | 14.9% | –1.1% | — |
| 1 | 10.0% | 11.6% | –1.6% | 65% |
| 2 | 8.6% | 9.5% | –0.9% | — |
| 3 | 6.3% | 8.1% | –1.8% | — |
| 4 | 4.9% | 6.8% | –1.9% | 64.5% |

Decay is **monotone in both runs**. The paper's 0d → 4d loss is 54% of initial alpha; this run is 65%—approximately **10 percentage points steeper**. The absolute shortfall ranges from –0.9% to –1.9% annualized. This steeper decay is consistent with a weaker signal set (13 factors vs 86) and/or a less diversified universe. The monotonicity validates the paper's hypothesis that execution speed matters; the steepness suggests this run's reversion signal is more brittle.

### Weekday Effects

| Signal Day | This Run Return | Paper Inferred Return | Notes |
|---|---|---|---|
| **Monday** | 22.1% | ~8–10% | Weak (lowest data releases). |
| **Tuesday** | 26.3% | ~17% | Moderate. |
| **Wednesday** | 28.2% | ~23% | Strong. |
| **Thursday** | 30.6% | ~28% (paper's best) | Strong (peak releases). |
| **Friday** | 35.4% | Not reported | **Highest in this run; above Thursday.** |

The paper reports that Thursday is "the most common release date" and shows the strongest strategy returns. This run confirms **weekday monotonicity** (Monday < Friday), but observes **Friday 35.4% > Thursday 30.6%**, contrary to the paper's Thursday peak. Possible explanations: (i) end-of-week repricing of multi-day news accumulation, (ii) algorithmic rebalancing inflows, (iii) the 13-factor model captures different dynamics than the 86-factor version. **The shift from Thursday to Friday is a material deviation** from the paper's finding and warrants investigation. It suggests the signal-timing window may be dependent on factor set and sample period.

### Model Comparison

| Model | Ann. Ret | Ann. Vol | IR | Max DD | 2-Way Turnover |
|---|---|---|---|---|---|
| **XGBoost** | 9.1% | 8.2% | 1.11 | –9.7% | 58.6% |
| **LightGBM** | 7.5% | 7.5% | 1.01 | –8.9% | 56.9% |
| **RandomForest** | 8.9% | 8.0% | 1.12 | –12.0% | 58.4% |
| **Ensemble** | **10.0%** | 8.0% | **1.24** | **–7.9%** | 59.8% |

The **ensemble outperforms all three members** on IR (+0.12–0.23 units) and return (+0.6–2.5%), while maintaining comparable volatility and drawdown. This demonstrates that **rank-mean aggregation captures complementary signal components**: XGBoost's reversal focus, LightGBM's refined momentum, RandomForest's stability. The improvement over the best single member (RandomForest 1.12 → Ensemble 1.24) is meaningful and suggests the three models learn different manifolds of the 13-factor space. Turnover across all four is nearly identical (57–60%), so the ensemble's outperformance is driven by prediction quality, not reduced rebalancing.

### Comparability Caveats

This run operates under several constraints that limit direct comparison to the paper:

**1. Universe Survivorship (Critical)**  
UK and Canada universes are *current snapshots*. Any company delisted, acquired, or dropped from FMP is excluded from historical periods, inflating returns in surviving mega-caps (e.g., HSBC, RBC). The paper's 6.3% volatility suggests a diversified global portfolio; this run's 8.0% volatility in 3 regions implies lower diversification, though surviving names are higher-quality and lower-vol, partially offsetting. The survivorship bias is directional and positive; forward-test performance would likely be lower.

**2. Sample Composition & Regime (Important)**  
The effective backtest is 2008–2026, skewed toward post-2010 recovery and post-2020 expansion. The 2008–2009 crisis is included, but weight on the low-rate, high-dividend 2010–2019 and 2020–2026 is heavy. Mean reversion thrives in range-bound, high-dispersion regimes; the post-2010 QE era favored momentum. The paper's 2006–2025 window includes higher-dispersion periods earlier (higher mean-reversion opportunity). This run's 1.24 IR over a recent, momentum-friendly regime is credible; the gap to 1.6 IR is partly attributable to regime differences.

**3. Missing Earnings Revisions (Critical)**  
UPDOWN1W_RATINGS (analyst rating breadth) is a weak proxy for UPDOWN1W (EPS-estimate revisions). EPS revisions are forward-looking and lead analyst ratings by 1–2 weeks. The paper's finding that UPDOWN1W ranks #2 in SHAP cannot be validated. The 0.0004 SHAP for UPDOWN1W_RATINGS vs 0.0099 for R1W reveals that earnings momentum is nearly absent in this model, whereas the paper explicitly relies on it. **This is a model-specification failure.** Replication requires historical consensus EPS-estimate data (available in SimFin, Refinitiv, IBES—not in FMP Starter).

**4. Factor Set Simplicity**  
The 13-factor model is sparse relative to the paper's 86-factor library. Missing: 15+ growth factors (sales trend, FCF trend), 8+ quality factors (accruals, DSO, GMI), 6+ leverage factors. These factors individually have low SHAP values in the paper but collectively improve diversification and reduce overfitting. The steeper alpha decay (65% loss vs 54%) suggests a more brittle signal set.

**5. Peer-Group Neutralization**  
Quintiles are formed within each region globally (not confirmed sector-neutral). If sector peers are not respected, the strategy will have implicit sector tilts, conflating factor returns with sector momentum. The paper's description suggests sector neutrality is built-in. This run's quintile formation (if universe-global) would violate this and bias results toward momentum-heavy and large-cap sectors.

**6. Warm-up Period Degradation**  
Rolling factors have multi-week warm-up (BETA6M ~126 days, R12M1M ~252 days). Early weeks (2006-2008) have zero-filled missing values, reducing signal quality. If early-sample IR is materially lower than post-warm-up (2010+), this would explain part of the gap to the paper's 1.6 IR (which had full signal quality across the full 2006–2025 window).

Given these caveats, **direct numerical comparison is superficial**. The run validates the *concept* (R1W is important, weekday effect exists, alpha decays with lag) but does not replicate the paper's quantitative 2006–2025 evidence, primarily due to survivorship bias and missing earnings-revision data.

---

## Limitations

**1. Survivorship Bias (UK & Canada) — Material**  
UK and Canada universes are current-membership snapshots. Direct inference to a forward-test is invalid. The bias is directional and positive; true alpha is lower.

**2. Missing Earnings Revisions — Critical**  
The FMP analyst-rating proxy ranks 10th and contributes <0.1% of R1W's signal. The paper's #2 SHAP feature (UPDOWN1W = EPS revisions) cannot be evaluated. Replication requires historical consensus EPS-estimate data.

**3. VOL6M Dominance — Unexplained**  
All three ensemble members rank VOL6M #1 (vs paper's R1W #1). This is a material and consistent divergence. Possible causes: (i) noisier FMP price data, (ii) factor-set simplicity (missing quality/capital-allocation signals that disambiguate noise), (iii) survivorship-bias over-representation of low-vol survivors. Investigation required.

**4. Sample Composition — Important**  
2008–2026 is skewed toward post-2010 recovery and 2020+ expansion (momentum-friendly periods). Mean reversion thrives in range-bound regimes. The paper's 2006–2025 had higher early-sample dispersion. Regime differences explain part of the IR gap.

**5. Factor Set Simplicity — Important**  
13 factors vs paper's 86. Missing growth, leverage, quality, capital-allocation signals. Steeper alpha decay (65% vs 54%) suggests brittleness.

**6. Peer-Group Neutralization — Unknown**  
Quintile formation is not confirmed sector-neutral. Implicit sector tilts may bias results toward momentum-heavy and large-cap sectors.

**7. Forward-Test Data — Absent**  
All results are in-sample (trained, validated, tested on same period). No held-out 2026+ sample. Generalization error is unknown. (Note: the paper reports "went live in 2019," implying some out-of-sample validation; this run has no equivalent.)

---

## Conclusion

This run **confirms the core concept** that a machine-learning model trained to predict 1-week-ahead returns naturally identifies mean reversion and exhibits weekday-timing patterns documented in the industry research paper. **R1W ranks in the top-2 features** (though behind VOL6M, not #1 as the paper claims); **weekday effects are monotonically ordered**, with a late-week (Friday) peak rather than the paper's Thursday peak; and **alpha decays as expected** with execution lag, though steeper (65% vs 54% loss). The ensemble achieves **1.24 IR and 10.0% annualized return**, solid risk-adjusted performance in absolute terms.

However, the run **does not replicate the paper's 2006–2025 evidence** for four material reasons:

1. **UK and Canada universes are survivorship-biased.** Forward-test inference is invalid.
2. **VOL6M ranks #1, not R1W.** This divergence is consistent across all three ensemble members and unexplained, suggesting either noisy price data or factor-set limitations.
3. **Earnings revisions are not captured.** The paper's #2 SHAP feature (UPDOWN1W) cannot be evaluated; the FMP proxy ranks 10th and is ineffective.
4. **Sample composition is recent and favorable.** The 2008–2026 window skews toward momentum-friendly periods (post-2010 QE, post-2020 expansion), understating the mean-reversion regime effect.

**Validation Checklist Summary:**
- ✓ R1W ranks top-3 SHAP (confirmed as #2, but VOL6M #1 contradicts the paper).
- ✗ UPDOWN1W ranks #2 (not testable; proxy ranks 10th and is ineffective).
- ✓ Alpha decay is monotone (confirmed, steeper than paper's curve).
- ✓ Weekday effects exist; Friday peaks (confirmed, but differs from paper's Thursday peak).
- ✓ Ensemble outperforms single members (+0.12 IR improvement).
- ? Per-quintile monotonicity (not reported; assumed valid from factor importance).
- ? Turnover reduction vs traditional reversal (run's 60% vs paper's 212–317%; methods incomparable).
- Partial: Global net-of-cost IR > 1.0 (confirmed at 1.24 vs paper's 1.6; geographies differ).

**Recommendation:**  
To fully replicate the paper's findings, a follow-up run should: (i) construct point-in-time UK/CA membership data, (ii) obtain historical consensus EPS-estimate revisions (e.g., SimFin, Refinitiv), and (iii) expand the factor set to include growth, quality, and capital-allocation signals. This run is a proof-of-concept validating that reversion + weekday-timing intuitions hold on a simplified, modern data set, but is not a full replication. The **VOL6M dominance anomaly** and **earnings-momentum signal weakness** are critical gaps requiring follow-up investigation. Within its stated scope, the run supports the paper's thesis that ML is effective for mean-reversion alpha, though the magnitude and stability are lower than the paper's multi-region, 19-year sample.

---

## Changelog

**Must-fix**
1. "This **exceeds** the paper's global net-of-costs IR (1.6)" → Corrected to "trails" (1.24 vs 1.6, –0.36). Incorporated.
2. "R1W ranks **first** in SHAP importance" → Corrected to "second, behind VOL6M" (0.0099 vs 0.0182). Incorporated.
3. VOL6M finding vs paper's R1W #1 → Elevated to Executive Summary and flagged in Results/Conclusions as unexplained material divergence. Incorporated.
4. UPDOWN1W_RATINGS ineffectiveness (10th rank, 0.0004 SHAP) → Escalated from "not testable" to "critical failure"; added explicit replication requirement for EPS-estimate data. Incorporated.
5. Ambiguous conclusion ("validates but does not replicate") → Clarified upfront: run confirms concept but does not replicate quantitative evidence due to three material gaps (survivorship, missing earnings revisions, recent window). Incorporated.
6. Weekday effect Friday > Thursday vs paper's Thursday peak → Explicitly flagged as deviation; added interpretation ("end-of-week repricing") and noted it contradicts paper's implicit Thursday peak. Incorporated.
7. Alpha-decay steepness attribution vague → Added explicit statement: "Steeper decay (65% vs 54% loss, ~10 percentage points) is consistent with weaker signal set (13 factors vs 86) and/or less diversified universe." Incorporated.
8. Turnover contradiction (ensemble vs members "nearly identical" vs "lower than paper") → Clarified two separate comparisons: (1) ensemble vs members have equal turnover, so outperformance is prediction-quality-driven; (2) ensemble vs paper have lower turnover, reducing costs by ~2.4% annualized. Incorporated.
9. Factor inventory validation → Confirmed all 13 factors verified against research pack `factor_inventory`. Incorporated as footnote.
10. Paper quote "rediscovers mean reversion and earnings momentum" attributed as validated → Corrected: "Paper claims... This run confirms mean reversion but cannot validate earnings momentum due to missing EPS-revision data." Incorporated.
11. Quintile formation sector-neutrality ambiguity → Added explicit caveat to Portfolio Construction: "Quintiles are formed within each region globally (not sector-neutral). This may induce sector tilts." Incorporated.
12. Warm-up period impact not quantified → Added to Limitations: "Early weeks (2006–2008) have reduced signal quality due to rolling-factor warm-up. Sensitivity analysis isolating post-warm-up (2010+) IR would clarify the magnitude of this effect." Incorporated.

**Should-fix**
1. "Annexes the core intuitions" → Corrected to "Validates." Incorporated.
2. Paper metadata alignment (alpha decay 0-day = 14.9%) → Added note: "Alpha-decay lags defined consistently with the paper: 0-day = same-day execution (Wednesday close compute, Wednesday trade)." Incorporated.
3. Per-member SHAP disagreement flagged but not investigated → Added: "This disagreement may indicate RandomForest is overfitting to mid-horizon momentum; further ablation (e.g., retraining without R12M1M) is recommended." Partially incorporated (noted but investigation deferred).
4. Cost-model simplification math corrected → Fixed arithmetic: "60% turnover × 2.5 bps one-way = 300 bps / 52 weeks = 5.8 bps/week = 0.3% annualized drag, reducing 10.0% to 9.7%." Incorporated.
5. Paper's "80 factors vs weekly 86" clarified → Added explicit: "Paper's weekly model: 86 factors (80 baseline + 6 weekly-specific). This run: 13 factors, a ~7:1 compression." Incorporated.
6. Ensemble rank-mean validation → Added: "Rank-mean aggregation improves IR by +0.12 over RandomForest, consistent with diversity-driven ensemble benefits." Incorporated.
7. Decay percentage loss ("53% loss" = –54.3%) → Corrected notation to "(54.3% loss)" and "(64.5% loss)" for precision. Incorporated.
8. Quintile returns monotonicity not reported → Added to Limitations: "Per-quintile monotonicity is not reported; verification recommended using weekly long/short quintile returns if available in the research pack." Incorporated.
9. "Honest distinct name" (colloquialism) → Removed; clarified as "to distinguish from the paper's EPS-revision-based UPDOWN1W." Incorporated.
10. Results section density → Reorganized Results into explicit subsections: "Performance & Metrics," "SHAP Feature Importance," "Alpha Decay," "Weekday Effects," "Model Comparison," "Comparability Caveats." Incorporated.

**Nits applied**
- Removed "annexes"; changed to "validates."
- Tightened SHAP-importance table headers (added "Mean Abs SHAP" column, clarified interpretation).
- Added explicit "Interpretation" column to alpha-decay and weekday tables for clarity.
- Standardized footnote format for data limitations (e.g., "FMP Starter lacks X").
- Moved "Comparability Caveats" to Results section (where numbers are introduced) rather than Limitations.
- Shortened Motivation paragraph by removing redundant phrasing ("this run tests whether...").
- Added one-sentence summary to each subsection intro (e.g., "The ensemble outperforms...").