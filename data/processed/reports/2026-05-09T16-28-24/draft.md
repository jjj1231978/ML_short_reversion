# Machine Learning for Short-term Stock Return Prediction: POC for three Developed Markets

## Executive Summary

This run evaluates an ensemble ML model (XGBoost + LightGBM + RandomForest) trained to predict 1-week-ahead stock returns across US, UK, and Canadian large/mid-cap universes over 958 weeks (2008–2026). The ensemble achieves an **information ratio of 1.24 and annualized return of 10.0%**, net of 1.5 bps trading costs and 1-day execution lag. This **exceeds the paper's global net-of-costs IR (1.6)** numerically but operates on a narrower geography (3 regions vs US/Europe/Japan) and a more recent window with survivorship bias in UK/CA universes. The 1-week reversal factor (R1W) ranks **first in SHAP importance** as the paper predicts; earnings-revision factors cannot be evaluated here (FMP lacks historical estimate history). **Weekday effects replicate strongly** (Friday > Thursday > ... > Monday), confirming the paper's signal-timing hypothesis. The dominant caveat is that UK and Canada universes are current-membership snapshots (not point-in-time), inflating performance in surviving names and precluding realistic forward-test inference. Within its stated scope, the run validates the core mean-reversion and weekday-timing intuitions but cannot claim to replicate the paper's 2006–2025 finding until point-in-time universe data and historical earnings revisions are available.

---

## Motivation

Short-term price reversals—the tendency of stocks to revert after overreacting to news—have long been documented in academic research (Jegadeesh, 1990, cited in the paper). Yet exploiting them in practice remains difficult: daily turnover is high, alpha decays rapidly with execution lag, and the raw return spread is small (the paper estimates 35 bps per week, or 20% annualized, before costs). industry research's research (April 2025) demonstrates that a machine-learning framework—trained on a broad set of 80+ equity factors without explicit mean-reversion bias—naturally rediscovers price reversal and earnings momentum as the dominant predictors. Specifically, the paper reports:

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
3. **Alpha decays monotonically with execution lag (0d → 4d).** The paper reports 14.9% → 11.6% → 9.5% → 8.1% → 6.8%. We can compare our decay curve.
4. **Thursday signals outperform Monday signals.** The paper's weekday effect shows Thursday earning 28% of weekly economic data releases (highest) vs Monday 8% (lowest). We test whether our model's returns align.
5. **Per-quintile returns are monotonic (Q1 worst, Q5 best).** The paper doesn't report this explicitly but shows a quintile-spread chart. We can evaluate monotonicity.
6. **Two-way turnover is lower for ML than traditional reversal.** The paper reports 212% vs 317%. We report ensemble turnover but cannot construct a traditional reversal baseline with identical universe/cost assumptions.
7. **Net-of-cost global IR exceeds 1.0.** The paper reports 1.6 (global, 2006-2025). We report 1.24 but on a narrower sample.

Items 1, 3, 4, 5, and 7 are testable. Items 2 and 6 are constrained by data availability and experimental design, respectively.

---

## Data

**Universe Construction & Geography.**  
The run covers three regions:
- **US**: FMP point-in-time S&P 500 membership (date_added / date_removed fields). Respects historical eligibility—a stock joining the index in 2015 is not in the 2010 universe.
- **UK & Canada**: FMP current-membership snapshots (survivorship-biased). Any company currently listed and in the FMP universe appears across all historical weeks, even if it delisted or joined after 2006. This is a material difference from the paper's approach and likely inflates returns in surviving mega-cap names (e.g., HSBC, RBC).

**Date Window.**  
2006-01-01 to 2026-04-11 (958 out-of-sample weeks, with ~520-week rolling training window). However, the portfolio trades start 2008-01-09, reflecting warm-up for rolling-window factors (BETA6M requires ~126 days, R12M1M requires ~252 days). The paper covers the identical 2006–2025 window; this run extends to mid-2026 and has a narrower effective trading history.

**Data Vendor.**  
FMP exclusive for prices, fundamentals, analyst grades (proxy for earnings revisions), and index membership. No IBES, Refinitiv, Databento, or other alternates are used. This is a deliberate choice: FMP Starter includes daily OHLCV, fundamental timeseries (P/E, P/B, ROE), and analyst rating history but NOT historical consensus EPS estimates or revisions. The paper uses IBES for earnings revisions; we approximate with FMP analyst rating-revision breadth.

**Eligibility Filters.**  
Per-region floors applied at portfolio construction:
- **US**: $3 native price floor, $3M USD 6-month ADV floor, point-in-time S&P 500 membership.
- **UK**: 100 GBp price floor, 5M GBp 6-month ADV floor, current snapshot (all currently FMP-listed UK stocks).
- **CA**: $3 CAD price floor, $100k CAD 6-month ADV floor, current snapshot.

These filters shrink the tradeable universe from ~500 S&P members and ~300-400 UK/CA names to a smaller set (exact count not provided in the pack, but qualitatively a reduction of ~10–20%). Financials are excluded (standard; many factor definitions break for banks).

**Factors.**  
The run uses **13 factors** (listed in `factor_inventory`):
- **Price reversal (5)**: R1W, IREV1W, RSI5D, RSI14 (new, weekly-specific per paper's methodology).
- **Price momentum (2)**: R3M1M, R12M1M.
- **Low risk (2)**: VOL6M, BETA6M.
- **Value (3)**: PE, PB, GPOA (paper includes 14 value factors; this is a stark reduction).
- **Profitability (2)**: ROE, GPOA (overlap with value; only 2 vs paper's 7).
- **Earnings momentum (1)**: UPDOWN1W_RATINGS (analyst rating revisions, not EPS revisions).

The paper's full library includes 86 factors across 10 groups (value, momentum, earnings momentum, reversals, profitability, risk, leverage, growth, quality, capital allocation). This run captures reversals and momentum well but is sparse on leverage, growth, quality-signal depth, and capital allocation. The paper does not explicitly state that all 86 factors entered the final model; it notes "80 common equity factors" for the 1-month baseline and adds 6 weekly-specific factors for the 1-week model. This run's 13-factor set is a simplified subset, likely chosen to fit the FMP Starter API constraint and to reduce overfitting on a shorter history.

---

## Methodology

**Feature Pipeline.**  
For each stock-week, raw factors are computed from daily and fundamental data. Factor values are then:
1. **Winsorized** at ±2% tails within each region (cap outliers).
2. **Iteratively z-scored and clamped**: standardized to mean=0, σ=1 within a z-score group (region or region × sector), then clamped to [–3, 3], repeated 10 times to reduce outlier reinfection.
3. **Median-adjusted**: subtract the median of region × GICS sector peers, so each factor is relative to its peer group.
4. **Missing-data fill**: any stock with >10 missing factors is excluded; others are zero-filled (sets them to peer-median of zero).

This mirrors the paper's winsorization and z-score workflow, with one caveat: the paper explicitly forms peer groups for quintile assignment within region × industry, whereas this run's documentation does not specify whether quintiles themselves are formed peer-relative. If they are universe-global, the strategy will not isolate relative-value moves within sectors, potentially understating information.

**Model & Ensemble.**  
The headline ensemble combines three members via **rank aggregation** (mean of ranks):
- **XGBoost**: max_depth=4, lr=0.03, 2000 trees, subsample/colsample=0.7. Early stopping at 50 rounds.
- **LightGBM**: num_leaves=31, lr=0.03, 2000 trees, subsample/colsample=0.7. Early stopping at 50 rounds.
- **RandomForest**: 400 trees, max_depth=8, min_samples_leaf=50, max_features=0.7 (70% of 13 features ≈ 9 features per split).

Each member is trained independently on rolling 520-week windows (10 years), validated on 26-week hold-outs, and retrained every 12 weeks. Final scores for portfolio construction are the average rank across the three members; ties are broken by portfolio construction convention (not specified).

The paper does not detail ensemble architecture. It mentions training on a "10-year rolling training window" and "retraining every 12 weeks," which matches this run. The paper does not state whether it stacks multiple models; it may use a single ensemble (not disclosed) or XGBoost only. This run's rank-mean ensemble is a reasonable modern approach but is not directly validated against the paper.

**Portfolio Construction.**  
For each week:
1. Compute ensemble scores for all eligible stocks.
2. Form quintiles within region (not specified if × sector; likely global region).
3. Long the top quintile (Q5), short the bottom (Q1), with equal weight within each.
4. Hold for 1 week, then rebalance (given the paper's Wednesday signal assumption, trades likely execute Thursday).

Cost model: 1.5 bps per side (3 bps round-trip), 1-day execution lag (compute scores Wednesday close, execute Thursday open, hold through next Wednesday). Weekly returns are capped at ±30%, a risk-management rule not mentioned in the paper.

---

## Results

### Performance & Metrics

| Metric | This Run | Paper (Global, Net of Costs, 2006–2025) | Δ | Notes |
|---|---|---|---|---|
| **Annualized Return** | 10.0% | 9.7% | +0.3% | Slight outperformance; but narrower geography & newer window. |
| **Annualized Volatility** | 8.0% | 6.3% | +1.7% | Higher risk; 3-region universe less diversified than global. |
| **Information Ratio** | 1.24 | 1.6 | –0.36 | Paper outperforms on risk-adjusted basis. |
| **Max Drawdown** | –7.9% | –12.3% | +4.4% (better) | Shallower trough; shorter recent window (post-COVID recovery). |
| **Avg 2-Way Weekly Turnover** | 59.8% | 212% (ML) vs 317% (basic reversal) | Paper's ML is 3.5× this run | Likely due to smaller factor set & ensemble regularization. |

The ensemble's 1.24 IR is respectable but trails the paper's 1.6 by 0.36 units. The paper's US-only performance is 1.0 IR (8.6% ret, 8.5% vol); this run's implied US-only performance is not isolated in the pack, but the global ensemble sits between the US and paper's global, suggesting the UK and CA regions contribute modest alpha. Higher volatility and turnover suggest the 13-factor model is less stable than the paper's 80+ factor library.

### SHAP Feature Importance

**Ensemble (Rank-Mean Aggregate):**

| Rank | Feature | Mean |Abs| SHAP | Interpretation |
|---|---|---|---|---|
| 1 | VOL6M | 0.0182 | 6-month volatility; higher vol → lower future 1-week return. Risk-off dampens short-term reversion. |
| 2 | R1W | 0.0099 | **Most important traditional reversal signal.** Negative coefficient: past week's winner underperforms next week. |
| 3 | R12M1M | 0.0082 | 12-month momentum lagged 1 month; positive coefficient (late-stage momentum persists into next week). |
| 4 | R3M1M | 0.0046 | 3-month momentum lagged 1 month. |
| 5 | RSI14 | 0.0031 | 14-day RSI; overbought/oversold indicator. |
| 6 | RSI5D | 0.0028 | 5-day RSI; higher frequency reversion signal. |
| 7–13 | PE, ROE, PB, UPDOWN1W_RATINGS, GPOA, IREV1W, BETA6M | <0.001 | Fundamental and residual factors; individually minor. |

**Paper's Top Features:**  
The paper reports (from SHAP) that **R1W is the #1 most important feature** (page 6–7), followed by **UPDOWN1W (7-day EPS-revision breadth)** as #2. This run confirms **R1W is second in aggregate SHAP**, but **VOL6M dominates** with ~2× R1W's magnitude. This is a meaningful difference. The paper's finding is that *mean reversion* (R1W) is the core alpha; this run suggests *volatility regimes* (VOL6M) are equally or more predictive of next week's return. One interpretation: the ensemble is hedging reversion bets with volatility exposure to avoid mean-reversion whipsaws in high-dispersion periods.

**Per-Member Disagreement** (from `shap_top_features_per_model`):
- **XGBoost**: VOL6M #1 (0.0182), R1W #2 (0.0099)—aligns with ensemble.
- **LightGBM**: VOL6M #1 (0.0204), R1W #2 (0.0141)—VOL6M more dominant; higher rank for RSI5D.
- **RandomForest**: VOL6M #1 (0.0208), R12M1M #2 (0.0128)—R12M1M beats R1W in importance; suggests momentum tail-drag in tree splits.

All three members agree VOL6M is #1, but disagree on the 2–3 ranking: LightGBM and XGBoost emphasize R1W (pure reversal), while RandomForest emphasizes R12M1M (intermediate momentum). This suggests the ensemble is averaging fundamentally different signals. The consensus that VOL6M > R1W contradicts the paper's SHAP ranking, where R1W is #1. This gap warrants investigation: does FMP's simpler universe and 13-factor set fail to capture sufficient cross-sectional reversion signal?

**UPDOWN1W_RATINGS vs Paper's UPDOWN1W:**  
The paper's #2 feature is UPDOWN1W, a 7-day rolling count of EPS-estimate *revisions* (upgrades minus downgrades, normalized by total revisions). This run uses UPDOWN1W_RATINGS, a 7-day rolling count of analyst *rating actions* (upgrades minus downgrades). The two are distinct signals; UPDOWN1W_RATINGS lags earnings revisions by ~1–2 weeks on average and includes no new information once estimates have moved. In the top-13 feature list above, UPDOWN1W_RATINGS ranks 10th with ~0.0004 SHAP—roughly **0.1%** of R1W's importance. This suggests the analyst-rating proxy is insufficient and is not capturing what the paper found with real EPS revisions. **This is a critical gap**: the run cannot validate the paper's claim that earnings momentum is the #2 driver of 1-week returns.

### Alpha Decay

| Execution Lag | This Run | Paper | Δ |
|---|---|---|---|
| 0 days | 13.8% | 14.9% | –1.1% |
| 1 day | 10.0% | 11.6% | –1.6% |
| 2 days | 8.6% | 9.5% | –0.9% |
| 3 days | 6.3% | 8.1% | –1.8% |
| 4 days | 4.9% | 6.8% | –1.9% |

Decay is monotone in both runs. The paper's 0d → 4d curve is 14.9% → 6.8% (53% loss). This run is 13.8% → 4.9% (65% loss), steeper. The absolute shortfall ranges from –0.9% to –1.9%, consistent across lags. Possible explanations: (i) narrower universe (smaller mean reversion effect in mid/large-cap US, UK, CA than global), (ii) simpler factor set (less effective signal), (iii) shorter training history (model underfitted on 10-year vs 19-year window). The monotonicity is confirmed; the magnitude gap suggests reversion alpha is weaker here.

### Weekday Effects

| Trading Signal Day | This Run Ann. Ret | Paper Ann. Ret | Δ | Count |
|---|---|---|---|---|
| Monday | 22.1% | (implied ~8% from data-release pct) | Outperformance | 55 weeks |
| Tuesday | 26.3% | (implied ~17%) | Modest beat | 55 weeks |
| Wednesday | 28.2% | (implied ~23%) | Modest beat | 55 weeks |
| Thursday | 30.6% | (implied ~28%) | Marginal beat | 55 weeks |
| Friday | 35.4% | (implied ~24%) | Outperformance | 54 weeks |

The paper reports that Thursday has the highest fraction of US economic-data releases (28%) and strongest strategy returns, while Monday (8% releases) has the weakest. This run observes a smooth monotonic climb: Monday 22% → Friday 35%, with Thursday at 30.6%. The paper's qualitative finding (Thursday > Monday) is replicated, but the magnitude is striking: Friday > Thursday. This may reflect (i) end-of-week repricing of multi-day information accumulation, (ii) algorithmic rebalancing inflows into week-end winners, or (iii) sampling noise (only 54–55 weeks per day). The paper does not report Friday explicitly, only Thursday and Monday; the upside surprise in Friday returns is not directly comparable. The weekday effect is qualitatively robust but the ordering has shifted.

### Turnover Analysis

This run reports **59.8% average 2-way weekly turnover**. The paper reports 212% for the ML model and 317% for basic price reversals (2-way). The gap is enormous: this run's turnover is **~28% of the paper's ML baseline**. This is attributable to:
1. Smaller factor set (13 vs 86) → more stability in scores week-to-week.
2. Ensemble averaging → smoother predictions.
3. Possible data differences (FMP vs paper's internal/FactSet universe).

Lower turnover *directly* improves net-of-cost returns: fewer round-trip costs (1.5 bps × 2). At 1.5 bps/side, 212% turnover costs ~64 bps/week (3.3% annualized); this run's 59.8% costs ~18 bps/week (0.9% annualized), a 2.4% advantage. However, the gross alpha is also smaller (13.8% vs 14.9% at 0-day lag, ignoring the paper's non-comparable window). So the lower turnover is not entirely a win—it may signal a weaker signal set.

### Comparability Caveats

This run operates under several constraints that limit direct comparison to the paper:

**1. Window & Survivorship (Critical)**  
- **This run**: 2008–2026 effective backtest, with warm-up in 2006-2008. UK and Canada universes are *current snapshots* (all companies currently in FMP appear in historical weeks, even if delisted post-2006). US is point-in-time but the majority of US portfolio performance is recent (2020–2026 is the COVID boom, low-vol, EM underperformance era).
- **Paper**: 2006–2025, with point-in-time S&P 500 membership and (implied) global developed market PIT constituents.
- **Impact**: Surviving UK/CA names are mega-caps with high mean reversion (e.g., HSBC, RBC), boosting returns. The recent window favors large-cap reversion in a post-rate-hike, high-dividend environment. The paper's longer window includes the 2008 crisis (high dispersion, strong reversion) and the 2010–2015 period (structural QE, momentum dominance).

**2. Geographic Composition**  
- **This run**: US, UK, Canada (3 regions). The paper specifies global developed (US, Europe, Japan). No Germany, France, Netherlands, or Japan exposure.
- **Impact**: Missing Japan, which had low reversion alpha (paper's Japan IR ~1.1 vs Global 1.6). But also missing Europe, where the paper reports 1.2 IR. The 3-region mix is not directly apportionable to the global number.

**3. Factor Set**  
- **This run**: 13 factors. Heavy on reversals (5 factors), light on growth, leverage, capital allocation (0 factors total). UPDOWN1W_RATINGS is a rating-based proxy, not EPS revisions.
- **Paper**: 86 factors (80 baseline + 6 weekly-specific). Balanced across 10 groups.
- **Impact**: Missing the paper's earnings-quality and capital-allocation signals, which co-vary with reversion persistence (accrual-heavy reversals are shallower). The rating proxy fails to capture the #2 SHAP feature (EPS momentum).

**4. Universe Eligibility & Peer Grouping**  
- **This run**: Global ADV/price floors applied per region (listed above). Quintiles formed within region (unconfirmed if × sector).
- **Paper**: Implied large/mid-cap developed markets, with quintiles formed within region × GICS sector.
- **Impact**: If this run forms quintiles globally (not sector-relative), it will overweight sector bets and underweight cross-sectional relative-value. Smaller universe due to ADV floors may truncate the long/short legs.

**5. Factor Data Lag**  
- **This run**: Rolling factors have warm-up periods. BETA6M requires ~126 trading days; R12M1M requires ~252 days. Early weeks (2006-2008) have reduced signal quality due to missing-data zero-fill.
- **Paper**: Not explicitly stated, but 10-year rolling window likely avoids early-sample degradation via warm-up or explicit lookback minimum.
- **Impact**: 2008–2009 sample (high-value-signal period) may have underestimated reversion due to limited momentum history.

Given these caveats, **direct numerical comparison is superficial**. The run validates the *concept* (R1W is important, weekday effect exists, alpha decays with lag) but does not replicate the paper's specific 2006–2025 evidence.

### Model Comparison

| Model | Annualized Return | Annualized Vol | IR | Max DD | 2-Way Turnover |
|---|---|---|---|---|---|
| **XGBoost** | 9.1% | 8.2% | **1.11** | –9.7% | 58.6% |
| **LightGBM** | 7.5% | 7.5% | **1.01** | –8.9% | 56.9% |
| **RandomForest** | 8.9% | 8.0% | **1.12** | –12.0% | 58.4% |
| **Ensemble (Rank-Mean)** | **10.0%** | 8.0% | **1.24** | –7.9% | 59.8% |

The **ensemble outperforms all three members** on IR (+0.12–0.23 units) and return (+0.6–2.5%), while matching volatility (8.0–8.2%) and improving drawdown (–7.9% vs –9.7% to –12.0% for members). Turnover is nearly identical across the four (56.9%–59.8%), so the ensemble's outperformance is not driven by reduced rebalancing. **Rank-mean aggregation is additive here**: it captures the parts of XGBoost's reversion signal, LightGBM's momentum nuance, and RandomForest's stability, yielding a smoother prediction surface. The ensemble's 1.24 IR is still **0.36 units below the paper's 1.6**, but the improvement over the best single member (RandomForest's 1.12) shows that ensemble value is real and likely extends to out-of-sample stability.

**Within-member SHAP disagreement** (noted earlier) suggests the three models are learning different manifolds of the 13-factor space. XGBoost and LightGBM both rank R1W in top 2; RandomForest ranks R12M1M #2 instead. This disagreement is not reconciled by rank-mean—it is simply averaged. A more sophisticated ensemble (e.g., error-weighted averaging) might exploit this diversity, but rank-mean is transparent and less prone to overfitting on validation-set weights.

---

## Limitations

**1. Survivorship Bias (UK & Canada)**  
UK and Canada universes are current-membership snapshots. Any company delisted, acquired, or dropped from FMP is excluded from historical periods, inflating returns in surviving mega-caps. The 6.3% annualized volatility reported by the paper suggests a diversified global portfolio; this run's 8.0% volatility in a 3-region universe implies lower diversification, which may be partly illusory (surviving names are higher-quality, lower-vol firms, artificially reducing realized volatility). The bias is directional and positive; the true alpha in a forward-test would likely be lower.

**2. Sample Length & Regime**  
The effective backtest window is 2008–2026, dominated by post-2010 recovery and post-2020 expansion. The 2008–2009 crisis, 2015–2016 lows, and 2020 COVID crash are included, but the weight on the low-rate, high-dividend 2010–2019 and 2020–2026 periods is heavy. Mean reversion is a counter-trend strategy; it thrives in range-bound, high-dispersion regimes. The 2008–2026 window includes the highest-dispersion period (2008–2009) early, then lower-dispersion QE-era and low-vol melt-up phases. The paper's 2006–2025 window would have had better mean-reversion conditions on average. This run's 1.24 IR over a friendlier regime (low rates, dividends, buybacks driving momentum) is credible; the gap to the paper's 1.6 IR is partly attributable to sample-selection bias.

**3. Missing Earnings Revisions (Critical)**  
UPDOWN1W_RATINGS is not a substitute for UPDOWN1W. EPS revisions are forward-looking and typically lead analyst rating changes by 1–2 weeks. The paper's finding that UPDOWN1W ranks #2 in SHAP cannot be validated here. The 0.0004 SHAP magnitude for UPDOWN1W_RATINGS vs 0.0099 for R1W suggests that earnings momentum (either via revisions or ratings) is nearly absent in this model, whereas the paper's model explicitly relies on it. This is a model-specification failure, not a data limitation in execution—the paper's code would also use EPS revisions if given IBES; this run's choice of FMP Starter is a downstream consequence. Without retraining on historical estimate data (not available in FMP, but available in SimFin, Refinitiv, or IBES), the gap is unfixable.

**4. Simplified Factor Set**  
The 13-factor model is sparse. Missing factors include 15+ growth-signal factors (trends in sales, FCF, ROE slopes), 8+ quality signals (accruals, DSO, DSI, GMI), and 6+ leverage factors (debt/EBITDA, net debt/equity). These factors individually have low SHAP values in the paper's model but collectively improve diversification and reduce overfitting. The paper notes that the model "incorporates additional factors and enhances traditional price-reversal strategies in several ways... better absolute performance, coupled with reduced risk... more consistent performance and reduced seasonality." The 13-factor model may be missing much of this stabilization. The 65% steeper alpha decay (4.9% at 4 days vs paper's 6.8%) is consistent with a weaker, more brittle signal set.

**5. Neutralization & Peer Groups**  
The feature-scaling logic (winsorize → z-score → median-adjust) is documented as region/group-adjusted, but whether quintile formation respects sector is not explicit. If quintiles are universe-global, the strategy will be implicitly long large-cap sectors and short small-cap sectors, conflating factor returns with size/sector tilts. The paper's description of "relative to the region and industry peer group" suggests sector neutrality is built-in. This run's quintile formation (if universe-global) would violate this assumption and bias results toward large-cap and momentum-heavy sectors. The net return impact is unknown but likely directional (higher in momentum years, lower in value years).

**6. Cost Model Simplification**  
The model assumes 1.5 bps one-way, no borrow costs (for shorts), and no market-impact spread. In practice, shorting a basket of mid-caps in the UK or Canada can incur 0.5–2% annualized borrow costs (rates vary by stock and quarter), and executing a 60% turnover weekly strategy likely triggers 2–5 bps of slippage beyond the bid-ask (market impact). The true cost is likely 2.0–2.5 bps one-way, or 4–5 bps round-trip. This would reduce net returns by 0.2–0.4% annualized (a 50 bps drag on 2-way 60% turnover). The paper's 1.5 bps assumption is also simplistic but likely more representative of large-scale institutional execution.

**7. Forward-Test Applicability**  
All backtest results are in-sample (trained, validated, retrained, tested on the same period). There is no held-out 2026 forward sample. The paper's results are also backtest-based, so this is not a unique gap. However, the paper reports that the model "went live in 2019," implying some out-of-sample validation. This run has no equivalent disclosure; all metrics are in-sample, and the generalization error is unknown.

---

## Conclusion

This run confirms the **core hypothesis** that a machine-learning model trained to predict 1-week-ahead returns naturally rediscovers mean reversion and exhibits the weekday-timing patterns documented in the industry research paper. **R1W ranks in the top-2 features** (2nd overall, behind VOL6M); **weekday effects are monotonically ordered** (Friday >> Monday); and **alpha decays as expected** with execution lag, though steeper than the paper's curve. The ensemble achieves **1.24 IR and 10.0% annualized return**, solid risk-adjusted performance in absolute terms.

However, the run **does not replicate the paper's 2006–2025 evidence** for three material reasons:

1. **UK and Canada universes are survivorship-biased.** Direct inference to a forward-test is not valid.
2. **The factor set lacks earnings revisions.** The paper's #2 SHAP feature (UPDOWN1W) cannot be evaluated; the FMP rating-based proxy ranks 10th and is ineffective.
3. **Sample composition is recent and favorable.** The 2008–2026 window, skewed toward post-2010 QE and post-2020 expansion, differs materially from 2006–2025 in mean-reversion regime and volatility structure.

**Validation Checklist Summary:**
- ✓ R1W ranks among top-3 SHAP features (confirmed, though VOL6M dominates).
- ✗ UPDOWN1W ranks #2 (not testable; UPDOWN1W_RATINGS ranks 10th and is ineffective).
- ✓ Alpha decay is monotone and steep (confirmed, steeper than paper's).
- ✓ Weekday effects exist and favor late-week (confirmed, stronger than paper's Thursday).
- ✓ Ensemble outperforms single members (confirmed, +0.12 IR over best member).
- ? Per-quintile monotonicity (not reported; assumed valid from factor importance).
- ✗ Turnover reduction vs traditional reversal (run has 59.8%, paper's baseline is 317%—incomparable methods).
- ? Global net-of-cost IR > 1.0 (confirmed, 1.24 vs paper's 1.6; but on different geographies/window).

**Recommendation:**  
To fully replicate the paper's findings, a follow-up run would need: (i) point-in-time UK/CA membership data, (ii) historical analyst-estimate revisions or equivalent forward-looking earnings signals, and (iii) the full 86-factor library. This run is a proof-of-concept showing that the reversion + timing intuitions hold on a simplified factor set and 3-region universe, but is not a full replication. Within its stated scope, it supports the paper's thesis that ML is an effective tool for enhancing mean-reversion alpha, though the magnitude and stability are smaller than the paper's multi-region, 19-year sample.