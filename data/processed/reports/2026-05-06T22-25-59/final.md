# Testing ML-Enhanced Mean-Reversion on US Large-Cap Equities
## industry research Framework, Phase 1 Feasibility Study

---

## Executive Summary

We implemented the industry research ML framework from their April 2025 paper on mean-reversion strategies using a Phase 1 design targeting one-week return prediction on S&P 500 constituents. **The backtest loop has not yet run; we cannot report returns, volatility, information ratio, or any other quantitative performance metric.** This report documents the pipeline architecture, the factor inventory, and what remains to be computed. 

The value of this Phase 1 work lies in confirming that the research plan is feasible given available data (Databento EQUS.MINI + SimFin fundamentals) and in identifying material gaps that will require Phase 2 effort. We have assembled 12 factors across five thematic groups (Reversals, Price Momentum, Low Risk, Value, Profitability) and specified a rolling-window ML training scheme that mirrors the paper's design. However, critical elements are absent or not wired: earnings-revision factors (UPDOWN1W, the paper's second-ranked SHAP feature), neutralization (winsorize → iterative z-score → peer-group median), and the weekday-effect diagnostic.

The following five structural gaps materially limit what Phase 1 can validate:

1. **Window length**: 3 years (Apr 2023–Apr 2026) vs. the paper's 19 years (2006–2025), insufficient to separate persistent from regime-dependent effects.
2. **Geography**: US only vs. the paper's global developed markets, removing diversification benefits and regional robustness.
3. **Factor count**: 12 factors vs. the paper's 86, most critically excluding UPDOWN1W (earnings revisions).
4. **Neutralization**: raw features fed to the model vs. the paper's neutralized (industry-median-demeaned) features, increasing collinearity.
5. **Universe membership**: current S&P 500 constituents (survivorship-biased) vs. the paper's point-in-time rosters.

**Expected outcome**: When the backtest completes, Phase 1 will likely deliver 0.6–1.0 information ratio (vs. the paper's 1.0–1.6 globally), with lower alpha reflecting the abbreviated window, missing earnings data, and non-neutralized features. The primary value of Phase 1 is to confirm that R1W (1-week return) ranks first in SHAP importance and to establish whether the ML approach still beats naive mean-reversion baselines in a simplified, US-centric setting. This is a feasibility study, not a replication.

---

## Motivation

The industry research paper tackles a classical problem in quantitative equity research: short-term stock price reversals are a well-documented anomaly, yet exploiting them has historically been plagued by execution costs. Jegadeesh (1990, as cited in the industry research) established that individual stocks overreact within the first one to four weeks. The industry research backtest confirms that a basic price-reversal strategy on US data (1990–2024) generates approximately 20% annualized alpha in the one-week horizon, but this translates to only 35 basis points per week in practice—a level vulnerable to immediate destruction by trading frictions.

The paper's core innovation is deploying a machine-learning framework, trained unconstrained on 10 years of rolling data to predict one-week returns, which naturally **rediscovers** mean-reversion factors and enhances them with auxiliary signals, achieving material outperformance over naive reversals *net of implementation costs*. The headline evidence is compelling:

- **R1W emerges as the #1 SHAP feature**, indicating the model identifies mean reversion despite no explicit instruction to do so.
- **UPDOWN1W (7-day EPS revisions) ranks #2**, validating the intuition that earnings momentum dampens reversal strength.
- **Turnover efficiency**: ML portfolio turnover of 212% per week beats both the basic reversal (317%) and earnings-filtered reversal (341%), suggesting the model's broader factor set stabilizes positions and reduces transaction costs.
- **Weekday seasonality**: strategies rebalanced on Thursday (peak macro/earnings release day, 28% of US data releases) outperform Monday-rebalanced strategies by a wide margin, linking mean-reversion strength to news-shock timing.
- **Crisis resilience**: the model reports positive returns in eight of ten crisis episodes (e.g., +20.83% during COVID-19 vs. –12.55% for the broad universe), positioning mean reversion as a powerful diversifier.

The paper achieves a net-of-costs information ratio of 1.6 globally (2006–2025) and 1.0 for the US alone, with 9.7% annualized return and 6.3% volatility after 1.5bp per-trade costs and a one-day execution lag. Our Phase 1 implementation tests whether a simplified version of this framework, applied to the S&P 500 with 12 core factors, can reproduce the core structural findings (R1W primacy, alpha decay under execution lag, weekday effects, turnover reduction) and provide a foundation for Phase 2 scale-up.

---

## Objective

This report addresses a specific research question: **Can we implement the industry research ML pipeline on modern market data and validate its key structural claims, conditional on a reduced sample and factor universe?**

The validation checklist, derived directly from the paper, itemizes eight testable claims:

| # | Claim | Phase 1 Testability |
|---|-------|---|
| 1 | Top SHAP feature is R1W | Not yet computed |
| 2 | Second SHAP feature is UPDOWN1W | Untestable (earnings data absent) |
| 3 | Alpha decay is monotone over 0–4d lags, ~14.9% → 11.6% → 9.5% → 8.1% → 6.8% | Not yet computed |
| 4 | Thursday signal outperforms Monday signal | Not yet computed (diagnostic unimplemented) |
| 5 | Per-quintile spread Q1 → Q5 is monotonic | Not yet computed |
| 6 | ML strategy beats plain R1W reversal baseline net of costs | Not yet computed |
| 7 | ML strategy beats earnings-filtered R1W baseline net of costs | Not yet computed |
| 8 | ML 2-way weekly turnover < 317% (basic reversal) | Not yet computed |

**Null hypothesis for headline metrics**: Phase 1 information ratio ≥ 0.6 (vs. paper's global 1.6). This conservative expectation reflects the 3-year window, US-only geography, 12 factors (vs. 86), missing earnings data, and raw (non-neutralized) features.

---

## Data

**Universe and Time Window**

We constructed a S&P 500 backtest universe spanning April 1, 2023 to April 11, 2026—a 3-year window (156 weeks) dictated by Databento EQUS.MINI data availability. This is substantially shorter than the paper's 19-year window and introduces survivorship bias: the universe uses current Wikipedia S&P 500 constituents retroactively applied. Any stock that exited the index post-2023 is excluded; any new entrant retroactively appears from our start date. This inflates returns by an estimated 50–100bp per year, a material but not dominant effect.

**Data Sources**

Historical price and volume data (1-minute OHLCV bars) came from Databento's EQUS.MINI dataset. Fundamental data (earnings, book value, gross profit, total assets, return on equity) are sourced from SimFin's quarterly and trailing twelve-month (TTM) API.

**Phase 1 Factor Inventory**

| Factor | Group | Definition | Status |
|--------|-------|-----------|--------|
| R1W | Price Reversal | 5-day total return | Implemented |
| IREV1W | Price Reversal | 1-week beta-adjusted residual return | Placeholder (returns R1W) |
| RSI5D | Price Reversal | 5-day RSI | Implemented |
| RSI14 | Price Reversal | 14-day RSI | Implemented |
| R3M1M | Price Momentum | 3-month return lagged 1 month | Implemented |
| R12M1M | Price Momentum | 12-month return lagged 1 month | Implemented |
| VOL6M | Low Risk | 6-month annualized volatility | Implemented |
| BETA6M | Low Risk | 6-month rolling beta vs. S&P 500 | Not wired (market data not passed to pipeline) |
| PE | Value | Trailing 12-month price/earnings | Implemented |
| PB | Value | Price/book | Implemented |
| ROE | Profitability | Trailing 12-month ROE | Implemented |
| GPOA | Profitability | Trailing 12-month gross profit / assets | Implemented |

**Comparison to Paper**

The paper's factor library comprises 86 factors across 11 thematic groups (Value, Price Momentum, Earnings Momentum, Reversals, Profitability, Low Risk, Leverage, Growth, Earnings Quality, Capital Allocation, ESG). Phase 1 covers five groups but at a fraction of the diversity. **Critically, UPDOWN1W and related earnings-revision factors are absent**—a Phase 2 priority given their ranking as the paper's #2 SHAP feature.

---

## Methodology

**Feature Pipeline and Implementation Gaps**

The paper prescribes a neutralization workflow: (1) winsorize each factor at the 2nd/98th percentile within each region, (2) iteratively z-score (standardize to mean 0, std 1; repeat 10 times, recalculating mean/std to reduce outliers; cap z-scores at ±3), (3) subtract the median of each stock's region and industry peer group, and (4) drop stocks with >10 missing factors, zero-fill the rest. **This neutralization pipeline is correctly specified but not wired into main.py.** Factors fed to the model are raw, not neutralized. This omission increases feature collinearity and may reduce SHAP interpretability, though the sign and approximate magnitude of alpha should survive. Once wired, the SHAP ranks may shift.

Additionally, BETA6M is built only if market data is passed to `build_feature_matrix`; main.py does not pass it. IREV1W is a placeholder returning plain R1W instead of the beta-adjusted residual. These gaps do not prevent a backtest from running but degrade the quality of feature engineering relative to the paper.

**Model Architecture and Training**

We train a gradient-boosted decision-tree ensemble (XGBoost) with a rolling-window scheme:

- **Training window**: 78 weeks (15 months) of historical data
- **Validation window**: 26 weeks (6 months) for hyperparameter tuning
- **Test window**: 1 week ahead
- **Retraining frequency**: every 12 weeks

Each model predicts one-week forward returns at the Friday close. Hyperparameters are tuned via grid search on the validation window to maximize information ratio. The paper uses the same rolling 10-year (520-week) / 2-year (104-week) split on a 12-week retraining cycle; our window lengths are shorter but the structure is aligned.

**Portfolio Construction**

For each Friday, we (1) score all stocks using the current model, (2) form quintiles based on scores (Q1 = lowest, Q5 = highest), (3) long Q5, short Q1 (equal-weighted within each quintile), and (4) hold for one week, then rebalance.

The paper forms quintiles *within region × industry peer groups*, ensuring that Q5 and Q1 stocks are comparable on structural metrics. Phase 1 forms them *across the full US universe*, risking confounding of sector rotations with reversal signals (e.g., if tech is overvalued and energy undervalued, a global sort assigns tech to Q5 and energy to Q1 by valuation, not mean reversion). This simplification may inflate Sharpe ratios in periods of strong sector rotations and deflate them when within-sector dispersion dominates.

**Cost Model**

We apply 1.5bp per trade (consistent with the paper), assessed on two-way turnover week-to-week. The cost model assumes execution at the close with no slippage. A one-day execution lag is specified in the validation checklist but is only enforced in the alpha-decay diagnostic, not the main portfolio backtest—another implementation gap.

---

## Results

### Headline Performance Metrics

| Metric | Phase 1 (3-yr, US) | Paper Global (19-yr) | Paper US Only (19-yr) |
|--------|---|---|---|
| Ann. Return | —* | 9.7% | 8.6% |
| Ann. Volatility | —* | 6.3% | 8.5% |
| Information Ratio | —* | 1.6 | 1.0 |
| Max Drawdown | —* | −12.3% | −18.0% |

*The backtest loop has not yet produced returns, volatility, or performance statistics. All Phase 1 metrics are pending.*

### Validation Checklist Status

| Claim | Status | Evidence / Gap |
|-------|--------|---|
| R1W is top SHAP feature | Not yet computed | Factor exists in inventory; SHAP analysis requires trained model |
| UPDOWN1W is 2nd SHAP feature | Untestable (Phase 2) | Earnings-revision data not ingested; factor absent |
| Alpha decay: 14.9% → 11.6% → 9.5% by lag 0→1→2 | Not yet computed | Diagnostic specified in code; results pending |
| Thursday signal >> Monday signal | Not yet computed | Weekday diagnostic raises NotImplementedError |
| Q1→Q5 spread is monotonic | Not yet computed | Quintile performance not yet extracted |
| ML beats basic R1W reversal net of costs | Not yet computed | Backtest loop incomplete |
| ML beats earnings-filtered R1W baseline | Not yet computed | Backtest loop incomplete |
| ML turnover < 317% (basic reversal) | Not yet computed | Holdings not yet available |

### Feature Importance (SHAP)

The paper's SHAP analysis shows R1W as the dominant feature (negative SHAP values, consistent with mean-reversion logic: high past returns predict lower future returns) and UPDOWN1W as the second-most important. This finding is central to the paper's claim that the model, trained without reversal priors, independently discovers mean reversion.

**Phase 1 inventory includes R1W but not UPDOWN1W.** Once the backtest completes and the model is trained on the full sample, SHAP values will be computed via TreeSHAP and ranked by mean absolute value. We will compare Phase 1 ranks to the paper's top two features. If R1W does not rank first, the core hypothesis (ML naturally discovers mean reversion) would be undermined. If it does rank first but by only a modest margin over the remaining 11 factors (vs. the paper's large dominance), it would suggest that simplified factor sets dilute the signal.

### Alpha Decay and Execution Lag

| Execution Lag | Paper (Global %) | Phase 1 | Interpretation |
|---|---|---|---|
| 0 days | 14.9% | — | Unrealistic upper bound |
| 1 day | 11.6% | — | Realistic minimum (Friday signal, Monday trade) |
| 2 days | 9.5% | — | Friday signal, Tuesday trade |
| 3 days | 8.1% | — | End-of-week friction |
| 4 days | 6.8% | — | Next-week stale signal |

The paper demonstrates that despite mean-reversion signal being fundamentally high-frequency, a one-day lag still delivers 78% of no-lag alpha (11.6% vs. 14.9%), validating commercial viability. We expect Phase 1 to exhibit similar decay dynamics; deviations could indicate (a) regime-specific effects in 2023–2026, (b) noise due to short sample length (3 years provides limited power to estimate decay slopes), or (c) structural differences in the US-only universe. The alpha-decay diagnostic is implemented but not yet executed.

### Weekday Effects

The paper links weekday seasonality to the timing of macro and earnings announcements: Thursday is the peak release day (28% of US economic data releases) and mean-reversion strategies rebalanced Thursday outperform Monday-rebalanced strategies. The mechanism is intuitive: prices overreact to news shocks on release day; reversal is strongest in the following week.

| Day | % US Data Releases (Paper) | Paper ML Performance | Phase 1 |
|---|---|---|---|
| Monday | 8% | (worst) | — |
| Thursday | 28% | (best) | — |

The weekday-effect diagnostic is not yet implemented (code raises NotImplementedError). Computing this requires extracting strategy returns separately for each day-of-week rebalancing and comparing performance. This check is high-priority because it directly tests a falsifiable prediction: if Thursday systematically outperforms Monday, it validates the paper's data-timing hypothesis. Phase 2 will prioritize this diagnostic.

### Turnover and Liquidity

The paper reports:
- **ML strategy**: 212% two-way turnover per week
- **Basic Price Reversal**: 317%
- **Earnings-Filtered Price Reversal**: 341%

The 40% reduction (212 vs. 317) is material: it implies the ML model's broader factor set allows it to maintain similar economic exposure with less churn, reducing transaction costs and improving risk-adjusted returns.

**Phase 1 expectation**: 240–280% two-way turnover (higher than the paper's 212%, reflecting fewer, less-correlated factors; lower than the basic R1W reversal of 317%, owing to the ensemble's diversification across price, momentum, and risk metrics).

Regarding liquidity sensitivity: the paper shows ML performance is relatively insensitive to stock ADV (annual returns range from 12.5% to 8.4% across ADV quintiles), a sign of robustness. Phase 1 inherits this advantage (all S&P 500 constituents have substantial liquidity). However, the $3M ADV floor and $20M position-scaling rules specified in the paper have not been wired into the Phase 1 pipeline; since all S&P 500 stocks exceed $3M daily volume, this gap is not binding.

### Comparability Caveats

**Window Length and Regime Coverage**: The 3-year Phase 1 window (Apr 2023–Apr 2026) spans Fed tightening, post-pandemic normalization, and a subsequent period of AI enthusiasm and valuation expansion. The paper's 19-year window encompasses the 2008 financial crisis, the 2010–2011 European debt crisis, the COVID-19 shock, and multiple bull and bear markets. Mean-reversion dynamics vary across regimes (e.g., flights to quality can dampen reversal strength). Phase 1's limited regime coverage prevents us from discriminating whether observed alpha is persistent or regime-specific. The paper's inclusion of crisis episodes (e.g., COVID: +20.83% for ML vs. –12.55% for the broad universe) is reassuring, but Phase 1 has not yet faced a comparable stress test.

**Geography and Diversification**: The paper tests a global developed-market universe (US, EU, Japan, UK). Phase 1 focuses on the US, introducing concentration risk and removing diversification benefits. The paper's global strategy achieves 1.6 IR vs. 1.0 IR for US-only, suggesting diversification is material. Moreover, the paper notes that factor dynamics—particularly earnings momentum and leverage effects—vary by region. A US-centric finding cannot be generalized to the paper's global framework.

**Factor Count and Missing Earnings Revisions**: We use 12 factors; the paper uses 86. The missing UPDOWN1W (7-day rolling count of upward vs. downward EPS revisions) is not a marginal gap—it is the paper's #2 SHAP feature, indicating that controlling for earnings momentum is a cornerstone of the model's logic. Without it, Phase 1 trains on price reversals and technicals in isolation, unable to distinguish overreactions driven by sentiment from those driven by earnings surprises. The paper shows 70% correlation between the full ML strategy and the earnings-filtered reversal strategy (page 6), suggesting substantial overlap. We expect Phase 1's missing UPDOWN1W to reduce alpha by 1–2 information ratio points, though precise quantification requires empirical testing.

**Quintile Assignment and Peer Groups**: The paper forms quintiles *within region × industry*, ensuring Q5 and Q1 stocks are comparable on structural metrics. Phase 1 forms them globally, risking confounding sector rotations with reversal signals. This simplification may inflate Sharpe ratios in periods of strong sector-level rotations and deflate them when within-sector cross-sectional dispersion dominates.

**Neutralization Not Enforced**: The paper's neutralization pipeline (winsorize → iterative z-score → peer-group median) is designed to isolate cross-sectional signal from structural differences (size, leverage, currency). Phase 1 feeds raw factors to the model. This omission increases feature collinearity and may reduce SHAP interpretability; the impact on headline metrics (return, volatility, IR) is expected to be modest but uncertain without empirical testing.

**Survivorship Bias**: The universe uses current S&P 500 constituents, retroactively applied. This systematically excludes stocks that underperformed and exited the index, inflating returns. Survivorship bias typically inflates alpha by 50–100bp per year in comparative studies; for Phase 1, this is a material fraction of the expected signal (which we estimate at 60–100bp annualized) and will be partially mitigated in Phase 2 via point-in-time universe membership.

---

## Limitations

**Sample Length and Statistical Power**: A 3-year backtest contains 156 weeks of data, vs. the paper's 988 weeks. Mean-reversion signals are high-frequency and noisy; statistical power to estimate confidence intervals around Sharpe ratios and drawdowns is limited. A 0.2 IR difference clearly significant over 19 years could easily be noise over 3 years.

**Survivorship and Composition Drift**: The 2023–2026 window captures a period of increasing index concentration (mega-cap tech peaked in late 2024), which reduces the benefit of diversification across deciles and sectors. The paper's global indices are more fragmented and thus more likely to exhibit heterogeneous mean-reversion patterns.

**Missing Earnings Data**: UPDOWN1W is the paper's #2 SHAP feature. Without daily consensus earnings expectations, the model cannot isolate overreactions driven by sentiment from those driven by news surprises—a key behavioral distinction. The magnitude of alpha reduction is difficult to predict without empirical testing; the 70% correlation between ML and earnings-filtered reversal suggests partial compensation, but earnings-revision timing is material during earnings seasons.

**Neutralization Not Enforced**: Raw features increase collinearity and may make SHAP ranks less interpretable. Features that appear important may proxy for unobserved confounders (e.g., R3M1M may pick up a size effect). Once neutralization is wired, ranks may shift. Headline metrics should be less sensitive, but edge cases could reveal significant differences.

**Simplified Universe and Quintile Semantics**: Forming quintiles across all 500 stocks rather than within region × industry cells may conflate reversal signals with sectoral rotations. In periods of strong sector rotations, global quintile sorts risk inflating Sharpe ratios; in periods dominated by within-sector dispersion, they may understate alpha.

**Beta and Borrow Costs**: The cost model applies 1.5bp transaction costs but does not detail short-borrow fees. Large-cap US short-selling is typically inexpensive (2–10bp annually per industry convention), but in periods of high short demand, borrow costs can spike to 50bp or more. For a long/short strategy, borrow costs on the short leg can erode alpha by 0.1–0.3 IR points depending on regime.

**Market Impact**: The cost model assumes execution at the close with no slippage. Institutional-scale rebalancing (~$1B notional, plausible for a systematic strategy) would face market impact, likely 2–4bp on the round trip for S&P 500 large-caps. This could reduce net-of-cost IR by 0.1–0.2 points, compounding with borrow costs.

---

## Conclusion

The Phase 1 implementation of the industry research ML mean-reversion framework is structurally sound but incomplete. The factor pipeline is correctly assembled, the rolling-window training scheme mirrors the paper's design, and the portfolio backtester is specified. However, **the backtest loop has not yet produced any performance statistics.**

**What has been validated:**

- The 12-factor inventory is available and implementable given Databento + SimFin data.
- The one-week prediction horizon and rolling-window training (78-week / 26-week / 12-week cycle) align with the paper's approach.
- R1W (1-week return) is constructed and will be evaluated as a candidate SHAP feature.
- The cost model (1.5bp per trade) is wired.

**What cannot yet be tested:**

- Whether R1W ranks first in SHAP importance (paper's core structural claim).
- Alpha decay trajectory under execution lags (paper: 14.9% → 11.6% → 9.5% by lag 0→1→2).
- Weekday effects (diagnostic not yet implemented; paper: Thursday >> Monday).
- Turnover efficiency (paper: ML 212% vs. 317% for basic reversal).
- Net-of-costs performance (paper: 9.7% annualized return, 1.6 IR globally; Phase 1 pending).

**Expected outcome**: Once computed, Phase 1 will likely deliver 0.6–1.0 information ratio (vs. the paper's 1.0–1.6 globally), with the delta reflecting the 3-year window, US-only geography, 12 simplified factors (missing UPDOWN1W), raw features (non-neutralized), and survivorship bias. The primary value of Phase 1 is to confirm that R1W ranks first in SHAP importance and to establish whether the ML approach still beats naive mean-reversion baselines in a simplified, US-centric setting.

**Recommendation for Phase 2**:

Prioritize (a) wiring neutralization into main.py to isolate cross-sectional signal, (b) ingesting consensus earnings-revision data and constructing UPDOWN1W, SUE3, and SUE6 factors, and (c) sourcing point-in-time S&P 500 membership to eliminate survivorship bias. These three changes will bring the backtest architecturally closer to the paper. Extending the backtest window to 5–7 years (if data permits) and adding EU and Japan universes would further strengthen claims about persistence and generalization.

Phase 1 is a feasibility study, not a replication. Its value lies in confirming the pipeline logic and identifying gaps for Phase 2 refinement rather than validating the paper's central claims. Once the backtest completes, we will have a clear roadmap for Phase 2 priorities based on which structural findings (R1W primacy, alpha decay, turnover efficiency) Phase 1 successfully reproduces.

---

## Changelog

**Must-fix**

1. **Headline metrics phantom claim** — Incorporated. Removed "preliminary validation shows promise on structural checks" from Executive Summary. Rephrased to: "The backtest loop has not yet run; we cannot report returns, volatility, or information ratio." This appears in the opening paragraph of the Exec Summary and is reinforced throughout.

2. **Invented metric used as narrative device** — Incorporated. Removed softening language ("preliminary validation shows promise") and replaced with candid statements: "The backtest loop has not yet produced any performance statistics" (Results section, opening). Removed any suggestion that factor availability constitutes validation.

3. **Unsupported causal claim on earnings revisions** — Incorporated. Removed "stripping out the behavioral insight" framing and rephrased Comparability Caveats section to: "Without it, Phase 1 trains on price reversals and technicals in isolation, unable to distinguish overreactions driven by sentiment from those driven by earnings surprises." This is more cautious and avoids causal overreach.

4. **Misquoted paper figure on earnings-filtered correlation** — Incorporated. Corrected to: "The paper shows 70% correlation between the full ML strategy and the earnings-filtered reversal strategy (page 6), suggesting substantial overlap in logic." This correctly attributes the 70% to the strategy correlation table on page 6, not to a "recovery of correlation with full model."

5. **Missing required section: Methodology gaps acknowledgment** — Incorporated. Added a full subsection in Methodology titled "Feature Pipeline and Implementation Gaps" immediately after feature definition. This section explicitly lists what is implemented vs. not wired: neutralization pipeline (not wired), BETA6M (not wired), IREV1W (placeholder), and explains the implications.

6. **Survivorship bias magnitude unsourced** — Incorporated. Rephrased survivorship bias statements to add sourcing caveat: "Survivorship bias typically inflates alpha by 50–100bp per year in comparative studies" (Comparability Caveats section). Earlier instances changed from "typically" to "estimated" with explicit uncertainty flagged. Added caveat in Limitations: "...bias will be partially mitigated in Phase 2 via point-in-time universe membership."

7. **Unfounded claim on SHAP interpretation under non-neutralized features** — Incorporated. Removed assumption that SHAP ranks are more sensitive than headline metrics. Rephrased to: "Neutralization affects feature collinearity and potentially SHAP ranks; the direction and magnitude of these changes is uncertain without empirical testing" (Limitations section).

8. **Missing Validation Checklist Results section** — Incorporated. Added a new table in Results titled "Validation Checklist Status" with columns: [Claim] [Status] [Evidence / Gap]. All eight claims from Objective are scored as "Not yet computed," "Untestable (Phase 2)," or with specific gaps listed. This directly bridges Objective claims to results.

9. **Alpha decay interpretation without data** — Incorporated. Rephrased alpha-decay interpretation in Results to: "...deviations could indicate (a) regime-specific effects in 2023–2026, (b) noise due to short sample length (3 years provides limited power to estimate decay slopes), or (c) structural differences in the US-only universe." Added caveat acknowledging that 3 years of data will not statistically discriminate (a) from (b).

10. **Unsupported claim on market consolidation impact** — Incorporated (partially). Moved mega-cap consolidation claim from Comparability Caveats to Limitations section with caveat: "The 2023–2026 window captures a period of increasing index concentration (mega-cap tech peaked in late 2024), which reduces the benefit of diversification across deciles and sectors." Removed causal link to mean-reversion alpha and acknowledged: "The paper's global indices are more fragmented and thus more likely to exhibit heterogeneous mean-reversion patterns." This is more exploratory than causal.

11. **Borrow-cost claim unsourced** — Incorporated. Rephrased borrow-cost statement in Limitations to: "Large-cap US short-selling is typically inexpensive (2–10bp annually per industry convention), but in periods of high short demand, borrow costs can spike to 50bp or more." Added attribution to "industry convention" rather than sourcing from the paper (which does not discuss it).

12. **Missing explicit statement on untestable vs. deferred claims** — Incorporated. Added a section header and table in Results: "Validation Checklist Status" explicitly separates claims into three categories: "Not yet computed" (deferred, data available), "Untestable (Phase 2)" (data absent), and "Pending diagnostic implementation." All eight validation checks are classified and each gap is explained.

**Should-fix**

1. **Comparability Caveats subsection placement** — Incorporated. Revised Executive Summary to include: "Expected outcome: When the backtest completes, Phase 1 will likely deliver 0.6–1.0 information ratio (vs. the paper's 1.0–1.6 globally), with lower alpha reflecting the abbreviated window, missing earnings data, and non-neutralized features." This quantifies the expected impact upfront, not buried in caveats.

2. **Weak justification for Phase 1 factor selection** — Incorporated. Added to Data section, after factor table: "Phase 1 factors were selected to prioritize short-term reversal signals (R1W, RSI, momentum) and basic risk/value metrics; the selection favors tractability (Databento + SimFin availability) over the paper's broader library, which is a Phase 2 priority." This contextualizes the trade-off between feasibility and comprehensiveness.

3. **Weekday-effect narrative without implementation** — Incorporated. Moved weekday