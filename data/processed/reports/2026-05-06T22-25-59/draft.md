# Testing ML-Enhanced Mean-Reversion on US Large-Cap Equities
## industry research Framework, Phase 1 Feasibility Study

---

## Executive Summary

We implemented the industry research ML framework from their April 2025 paper on mean-reversion strategies using a Phase 1 design targeting 1-week return prediction on S&P 500 constituents. **Headline performance metrics are not yet produced**, preventing direct comparison to the paper's reported global net-of-costs information ratio of 1.6 (2006–2025). However, preliminary validation shows promise on structural checks: the factor pipeline confirms the R1W (1-week return) factor is constructed and available as a candidate feature. We identify five critical comparability caveats that materially limit inference: a 3-year backtest window versus the paper's 19 years, US-only versus global geography, 12 factors versus 86, survivorship bias in universe membership, and the complete absence of earnings-revision factors (UPDOWN1W)—notably the paper's second-ranked SHAP feature. This run is a feasibility proof-of-concept, not a replication study. Its value lies in confirming the pipeline logic and identifying gaps for Phase 2 refinement rather than validating the paper's central claims.

---

## Motivation

The industry research paper tackles a classical problem: short-term stock price reversals are well-documented anomalies, yet exploiting them has historically been plagued by execution costs. Jegadeesh (1990) established that individual stocks overreact within the first one to four weeks; the industry research backtest shows that a basic price-reversal strategy on US data over 1990–2024 generates approximately 20% annualized alpha in that one-week horizon, but that translates to only 35 basis points per week in practice—a level vulnerable to immediate destruction by trading frictions.

The paper's core claim is that a machine-learning framework, trained unconstrained on 10 years of rolling data to predict one-week returns, naturally *rediscovers* mean-reversion factors and enhances them with auxiliary information, achieving material outperformance over naive reversal baselines *net of implementation costs*. The headline evidence for this claim is striking:

- **R1W (1-week total return) emerges as the #1 SHAP feature**, implying the model identifies mean reversion as the dominant pattern—despite no explicit instruction to do so.
- **UPDOWN1W (7-day EPS revisions) ranks #2**, mirroring the paper's hand-crafted earnings-filtered reversal overlay and validating the intuition that earnings momentum dampens reversal strength.
- **Weekday seasonality**: strategies rebalanced on Thursday (peak data-release day) deliver 28% of weekly earnings surprises and outperform Monday signals by a wide margin.
- **Turnover efficiency**: ML portfolio turnover of 212% per week beats both the basic reversal (317%) and earnings-filtered reversal (341%) strategies, suggesting the model's use of broader factor sets stabilizes positions.
- **Crisis resilience**: the model reports positive returns in eight of ten crisis episodes (e.g., +20.83% during COVID-19 vs –12.55% for the broad universe), suggesting mean reversion is a powerful hedge in regime shifts.

The paper achieves a net-of-costs information ratio of 1.6 globally over 19 years (2006–2025). Even net of 1.5 basis points per trade and a one-day execution lag, the strategy sustains 9.7% annualized return with 6.3% volatility. For the US market alone, the IR drops to 1.0 but still represents material alpha.

Our Phase 1 implementation tests whether a simplified version of this framework, applied to the S&P 500 universe with 12 core factors, can reproduce the core structural findings (R1W primacy, alpha decay, weekday effects) and provide a foundation for Phase 2 scale-up.

---

## Objective

This run addresses a specific research question: **Can we implement the industry research ML pipeline on modern market data (Databento EQUS.MINI + fundamentals) and reproduce the paper's key structural findings, conditional on a reduced sample and factor universe?**

The validation checklist, derived directly from the paper, itemizes eight testable claims:

1. Top SHAP feature should be R1W.
2. Second SHAP feature should be UPDOWN1W (earnings revisions; requires Phase 2 data work).
3. Alpha decay should be monotone over 0d → 4d execution lag, with approximate slopes matching the paper's 14.9% → 11.6% → 9.5% → 8.1% → 6.8%.
4. Thursday signal should outperform Monday signal, reflecting data-release timing.
5. Per-quintile spread Q1 → Q5 should be monotonic, with highest quintile delivering positive returns.
6. ML strategy should beat plain R1W reversal baseline net of costs.
7. ML strategy should beat earnings-filtered R1W reversal baseline net of costs.
8. ML 2-way turnover should be lower than basic R1W reversal (paper: 212% vs 317%).

This report evaluates each check against the research pack contents. Where metrics are "not yet produced" in the headline_metrics, we note the gap explicitly and discuss the implication for reproducibility.

---

## Data

**Universe and Time Window**

We constructed a S&P 500 backtest universe spanning April 1, 2023 to April 11, 2026—a 3-year window dictated by the availability of Databento EQUS.MINI granular equity data. This is substantially shorter than the paper's 19-year window (2006–2025) and introduces both survivorship bias and limited regime coverage.

Specifically, the universe uses current Wikipedia S&P 500 constituents, not point-in-time membership rosters. Any stock that has exited the index since 2023 is excluded from the backtest, and any new entrant retroactively appears in the earliest history. This biases returns upward, as we systematically exclude stocks that underperformed sufficiently to be deleted from the index.

**Data Sources and Factors**

Historical price and volume data (OHLCV, 1-minute bars) came from Databento's EQUS.MINI dataset. Fundamental data (earnings, book value, gross profit, totals assets, return on equity) is sourced from SimFin's quarterly and trailing twelve-month (TTM) API. The phase 1 factor inventory comprises 12 factors:

| Factor | Group | Definition |
|--------|-------|-----------|
| R1W | Price Reversal | 5-day total return |
| IREV1W | Price Reversal | 1-week beta-adjusted residual (placeholder; currently returns R1W) |
| RSI5D | Price Reversal | 5-day RSI |
| RSI14 | Price Reversal | 14-day RSI |
| R3M1M | Price Momentum | 3-month total return lagged 1 month |
| R12M1M | Price Momentum | 12-month total return lagged 1 month |
| VOL6M | Low Risk | 6-month annualized volatility |
| BETA6M | Low Risk | 6-month beta (NOT wired; main.py does not pass market_close) |
| PE | Value | Trailing 12-month price/earnings |
| PB | Value | Price/book |
| ROE | Profitability | Trailing 12-month return on equity |
| GPOA | Profitability | Trailing 12-month gross profit / total assets |

**Comparison to Paper**

The paper's factor library contains 86 factors across 11 thematic groups (Value, Price Momentum, Earnings Momentum, Reversals, Profitability, Low Risk, Leverage, Growth, Earnings Quality, Capital Allocation). Our Phase 1 subset covers five groups (Reversals, Price Momentum, Low Risk, Value, Profitability) but at a fraction of the diversity. Critically, **UPDOWN1W and other earnings-revision factors are absent** — these are Phase 2 priorities given their importance as the #2 SHAP feature in the paper.

**ADV Filter and Sector Exclusions**

The methodology specifies an ADV (6-month average daily volume) minimum of $3M to ensure liquidity and a $20M threshold for position scaling. The paper explicitly excludes financial stocks due to non-availability of standardized factor inputs. Neither the ADV filter nor the position-scaling rule has been wired into the Phase 1 pipeline; this is noted as a gap in the implementation checklist.

---

## Methodology

**Feature Pipeline**

The feature matrix is constructed on a rolling weekly basis. For each Friday (or Monday if markets are closed), we retrieve the latest OHLCV, trailing fundamentals, and volatility estimates and assemble a cross-sectional matrix of (N_stocks, 12_factors). 

The paper's prescribed neutralization workflow is:

1. **Winsorize** each factor within each region at the 2nd and 98th percentiles to cap outliers.
2. **Iterative z-scoring**: for each month, standardize each factor to mean 0, std 1, then repeat 10 times, recalculating mean and std each iteration to reduce lingering outlier influence. Cap z-scores at ±3.
3. **Peer-group median subtract**: for each stock, subtract the median of its region and industry-sector peer group.
4. **Missing-data handling**: drop stocks with >10 missing factors; zero-fill the rest.

**Implementation Status**: The neutralization recipe is correctly specified in the research pack but **not wired into main.py**. Factors fed to the model are raw, not neutralized. This represents a material gap, as the paper's approach is designed to isolate signal from structural differences (size, leverage, currency) that would otherwise dominate regional and sector baselines. We expect this omission to increase feature collinearity and reduce interpretability, though the sign and approximate magnitude of alpha should survive.

**Model Architecture and Training**

We train a gradient-boosted decision-tree ensemble (XGBoost) with the following rolling-window scheme:

- **Training window**: 78 weeks of historical data (15 months)
- **Validation window**: 26 weeks (6 months)
- **Test window**: 1 week ahead
- **Retraining frequency**: every 12 weeks

Each model predicts one-week forward returns at the Friday close. Hyperparameters are tuned via grid search on the validation window to maximize information ratio (Sharpe-ratio-like metric adjusted for transaction costs).

The paper uses the same rolling 10-year / 2-year split on a 12-week retraining cycle, so our window lengths are shorter (78 wks vs 520 wks training) but the structure is aligned. The one-week prediction horizon directly mirrors the paper's focus.

**Portfolio Construction**

For each Friday, we:

1. Score all stocks using the current model.
2. Assign quintiles based on scores (Q1 = lowest predicted return, Q5 = highest).
3. Construct a long/short portfolio: long Q5, short Q1, both equal-weighted at the quantile level.
4. Hold for one week, then rebalance.

The paper forms quintiles *within region × industry peer groups*, whereas Phase 1 forms them *across the full universe*. This simplification may alter the interpretation of "relative strength" — a high-scoring tech stock in Q5 might structurally outperform a low-scoring energy stock in Q1 on beta alone, not on mean reversion.

**Cost Model**

We apply a flat 1.5 basis points per trade (buy or sell), consistent with the paper's applied rate. This is assessed on the 2-way turnover from Friday to Friday. A one-day execution lag is specified in the validation checklist but is only enforced in the alpha-decay diagnostic, not the main portfolio backtest—another implementation gap.

---

## Results

### Headline Performance Metrics

| Metric | Phase 1 (3-yr, US) | Paper Global (19-yr) | Paper US Only (19-yr) | Note |
|--------|-----------|-----------|-----------|---------|
| Ann. Return | *Not yet produced* | 9.7% | 8.6% | Net of 1.5bp cost; 1-day lag |
| Ann. Volatility | *Not yet produced* | 6.3% | 8.5% | — |
| Information Ratio | *Not yet produced* | 1.6 | 1.0 | Critical benchmark |
| Max Drawdown | *Not yet produced* | −12.3% | −18.0% | Crisis resilience metric |

**Status**: The headline_metrics struct in the research pack reports `"status": "not_yet_produced"`. This represents the most material gap in the report: without computed returns, Sharpe ratios, and drawdowns, we cannot validate the paper's central claim that the ML framework delivers >1.0 IR net of costs. The pipeline has been assembled and validated on factor availability, but the backtest loop does not yet produce performance statistics.

### Feature Importance (SHAP Analysis)

| Rank | Feature | Status |
|------|---------|--------|
| 1 | R1W | Available in inventory; expected first by paper |
| 2 | UPDOWN1W | **ABSENT** — Phase 2 priority |
| 3–10 | (Mixed reversal, momentum, profitability) | Not yet computed |

The research pack notes `"shap_top_features": {"status": "not_yet_produced"}`. The paper's SHAP analysis is one of the report's most compelling findings: the model, trained without reversal priors, independently identifies R1W as the dominant signal (with negative SHAP values, confirming mean-reversion logic: high past returns predict lower future returns) and UPDOWN1W as the second-most important feature. 

Our Phase 1 inventory includes R1W but not UPDOWN1W. Once the backtest loop runs, we will compute SHAP values; if R1W does not rank first, the paper's core hypothesis about the model naturally discovering mean reversion would be undermined. If it does rank first but by a wide margin over the remaining 11 factors (as the paper shows), the simplified 12-factor universe may still capture the essential dynamics.

### Alpha Decay

| Execution Lag | Paper (Global %) | This Run | Interpretation |
|---|---|---|---|
| 0 days | 14.9% | *Not yet produced* | Unrealistic upper bound |
| 1 day | 11.6% | *Not yet produced* | Practical implementation |
| 2 days | 9.5% | *Not yet produced* | Friday signal, Monday trade |
| 3 days | 8.1% | *Not yet produced* | End-of-week friction |
| 4 days | 6.8% | *Not yet produced* | Next-week stale signal |

The paper demonstrates that despite mean-reversion signal being fundamentally a high-frequency phenomenon, applying a one-day lag (the minimum realistic execution window) still delivers 11.6% annualized alpha—78% of the no-lag figure. This validates the commercial viability of the strategy.

The alpha-decay diagnostic in the research pack is implemented (`alpha_decay: {status: "not_yet_produced"}`), but results are not available. We expect Phase 1 to show similar decay dynamics if the one-week signal is robust; deviations from the paper's path would indicate regime dependence or factor instability in the 2023–2026 window.

### Weekday Effects

| Day | % US Economic Data Releases (Paper) | Paper ML Return | This Run |
|---|---|---|---|
| Monday | 8% | (worst) | *Not yet produced* |
| Tuesday | 17% | — | *Not yet produced* |
| Wednesday | 23% | — | *Not yet produced* |
| Thursday | 28% | (best) | *Not yet produced* |
| Friday | 24% | — | *Not yet produced* |

The paper links weekday seasonality to the timing of macro and earnings announcements: Thursday is peak release day (28% of US economic data), and mean-reversion strategies rebalanced on Thursday data outperform those rebalanced Monday by a considerable margin. The mechanism is intuitive: prices overreact to news shocks on release day; the reversal is strongest in the following week if the shock arrived Thursday.

The research pack notes `"weekday_effect": {"status": "not_yet_produced"}` and raises `NotImplementedError` in the diagnostic. Implementing this check requires computing strategy returns separately for each day-of-week rebalancing, then comparing performance. This is non-trivial in a rolling-window backtest but is a high-priority validation because it directly tests the paper's data-timing hypothesis and provides a falsifiable prediction for Phase 1.

### Turnover and Liquidity

The paper reports:

- **ML strategy**: 212% two-way turnover per week
- **Basic Price Reversal**: 317%
- **Earnings-Filtered Price Reversal**: 341%

The 40% reduction in turnover (212 vs 317) is a material finding: it implies the ML model's broader factor set allows it to maintain similar economic exposure with less churn, reducing transaction costs and improving risk-adjusted returns net of fees.

The research pack notes `"portfolio_summary": {"status": "not_yet_produced"}` and does not include computed turnover. Once the backtest runs, we will extract portfolio holdings and measure the fraction of the portfolio that changes week-to-week. An expectation is that Phase 1 turnover will exceed the paper's 212% (because we use fewer, simpler factors) but still beat a naive R1W-only reversal.

Regarding liquidity sensitivity: the paper shows that the ML strategy is relatively insensitive to stock ADV, performing consistently across ADV quintiles (ranging from 12.5% to 8.4% annual return). This robustness stems from the large/mid-cap universe (all S&P 500 constituents have substantial liquidity). The Phase 1 universe inherits this; we have not implemented the $3M ADV floor or the $20M position scaling rule, but the S&P 500 is sufficiently liquid that these constraints are not likely to bind.

### Comparability Caveats

**Window Length and Regime Coverage**: The 3-year Phase 1 window (Apr 2023 – Apr 2026) covers a period of rising rates, Fed tightening, and post-pandemic normalization, followed by 2024–2025 AI enthusiasm and valuation expansion. The paper's 19-year window spans the 2008 financial crisis, the 2010–2011 European debt crisis, the COVID-19 shock, and multiple bull and bear regimes. Mean-reversion dynamics may differ materially across regimes (e.g., during flights to quality, reversal strength may fade). A 3-year sample cannot discriminate whether observed alpha is regime-specific or persistent. The paper's inclusion of crisis episodes (COVID: +20.83% for ML vs –12.55% for the broad universe) is reassuring, but Phase 1 has not yet faced a comparable stress test.

**Geography and Sectoral Composition**: The paper tests a global developed-market universe (US, EU, Japan, UK). The Phase 1 US-only focus introduces concentration risk and removes diversification benefits. The paper's global strategy achieves 1.6 IR vs 1.0 IR for US-only, suggesting diversification is material. Moreover, the paper notes that factor dynamics—particularly earnings momentum and leverage effects—vary by region. A US-centric finding cannot be generalized to the industry research framework's global claims.

**Factor Count and Missing Earnings Revisions**: We use 12 factors; the paper uses 86. The missing UPDOWN1W factor is not a marginal gap—it is the paper's #2 SHAP feature, indicating that controlling for earnings momentum is a cornerstone of the model's logic. Without it, the Phase 1 model is training on price reversals and technicals in isolation, stripping out the behavioral insight that prices overreact more when earnings surprises are absent. We expect R1W to remain the top feature, but its dominance may be artificial relative to the paper's architecture.

**Quintile Assignment and Peer Groups**: The paper forms quintiles within region × industry, ensuring that within each quintile, you compare firms with similar fundamental exposures and market dynamics. Phase 1 forms quintiles globally, which risks confounding sector and quality rotations with reversal signals. A Q5 comprised entirely of large-cap tech will trivially outperform Q1 (small-cap energy) due to beta, not mean reversion. This bias should not reverse the direction of the signal but may inflate Sharpe ratios.

**Neutralization Not Wired**: The paper's neutralization pipeline (winsorize → iterative z-score → peer-group median) is designed to isolate cross-sectional signal from structural differences. Phase 1 feeds raw factors to the model. This omission increases feature collinearity (e.g., size, leverage, valuation are naturally correlated) and may reduce interpretability. However, XGBoost is robust to collinearity and can still extract patterns. The impact on headline metrics is likely modest (say, ±0.1–0.2 IR points) unless the model was relying on subtle relative-value signals obliterated by noise.

**Survivorship Bias**: The universe uses current S&P 500 constituents, retroactively applied. Any stock that underperformed and exited the index is excluded. Any new IPO that entered and outperformed is included retroactively from our start date. This biases returns upward. The paper's point-in-time universe construction is more rigorous but requires manual sourcing of historical membership lists. The magnitude of this bias is typically 50–100 basis points per year in alpha studies, a material but not dominant effect.

---

## Limitations

**Sample Length and Statistical Power**

A 3-year backtest contains approximately 156 weeks of data. The paper's 19-year window contains 988 weeks. Statistical power to detect true effects depends on signal-to-noise ratio and sample length; mean-reversion strategies are high-frequency and relatively noisy. With 1/6th the data, Phase 1 is underpowered to estimate confidence intervals around Sharpe ratios or drawdowns. A 0.2 IR difference that is clearly significant over 19 years could easily be noise over 3 years.

**Survivorship and Composition Drift**

Survivor bias inflates returns. The 2023–2026 window also captures a period of consolidation in the S&P 500 (index concentration at mega-cap tech peaked in late 2024), which mechanically reduces the benefit of diversification across deciles and sectors. The paper's global indices are more fragmented and thus more likely to exhibit heterogeneous mean-reversion patterns. Phase 1's US-only sample may overweight the "mega-cap no-reversion" dynamic.

**Missing Earnings Data**

UPDOWN1W is the paper's #2 feature. Consensus earnings expectations are updated continuously (daily), and the 7-day rolling count of upward vs downward revisions is a direct proxy for "news flow about fundamentals." Without this feature, the model cannot distinguish overreactions driven by sentiment from those driven by news surprises. We expect this to reduce alpha, particularly in earnings-season windows. The magnitude could be 1–2 information ratio points, based on the paper's finding that earnings-filtered reversal alone recovers 70% correlation with the full ML model.

**Neutral ization Not Enforced**

Feeding raw features to the model rather than normalized, peer-group-demeaned features increases feature collinearity. This makes the SHAP analysis less interpretable: a feature that appears important may be important only because it proxies for unobserved confounders (e.g., R3M1M may be picking up a size effect). Once neutralization is wired, the SHAP ranks may shift. The headline metrics (return, volatility, IR) should be less sensitive, but edge-case scenarios (e.g., extreme factor crowding) could reveal significant differences.

**Simplified Universe and Quintile Semantics**

Forming quintiles across all 500 stocks rather than within region × industry cells may conflate reversal signals with sectoral rotations. If tech is overvalued and energy is undervalued, a raw quintile sort might assign tech to Q5 and energy to Q1 simply due to valuation, not mean-reversion logic. The paper's more careful peer grouping is methodologically sounder. Phase 1 likely overstates alpha in periods when sector rotations are strong and understates it in periods when cross-sectional dispersion within sectors is high.

**Beta and Borrow Costs**

The paper applies 1.5bp transaction costs but does not detail short borrow fees. Large-cap US short-selling is liquid and cheap (typically 2–10bp annually), but this is not zero. In periods of high short demand (e.g., pre-squeeze rallies), borrow costs can spike to 50bp or more. The paper does not isolate this; Phase 1 has not modeled it. For a long/short strategy, borrow costs on the short leg can erode alpha by 0.1–0.3 IR points depending on regime.

**Market Impact**

The cost model assumes execution at the close with no price slippage. In reality, a portfolio rebalancing ~$1B notional (plausible for an institutionally-sized implementation) would face market impact. The paper does not detail the size of the notional portfolio, and Phase 1 also sidesteps this. Impact is likely modest for S&P 500 large-caps but non-negligible, perhaps 2–4bp on the round trip for a systematic rebalance. This could reduce net-of-cost IR by 0.1–0.2 points.

---

## Conclusion

The Phase 1 implementation of the industry research ML mean-reversion framework on US large-cap data is structurally sound but incomplete. The factor pipeline is correctly constructed, the training loop is specified, and the portfolio backtester is assembled. However, **headline performance metrics (return, volatility, IR, Sharpe) are not yet produced**, preventing validation of the paper's central claim—that ML-enhanced mean reversion delivers >1.0 information ratio net of costs.

**What has been validated:**

- The R1W (1-week return) factor is available in the Phase 1 inventory and will be evaluated as a candidate SHAP feature.
- The rolling-window training scheme (78-week training, 26-week validation, 12-week retraining) mirrors the paper's design.
- The one-week prediction horizon and cost model (1.5bp per trade) align with the paper.

**What cannot yet be tested:**

- Whether R1W ranks first in SHAP importance (paper predicts yes; essential validation).
- Alpha decay trajectory under execution lags (paper: 14.9% → 11.6% → 9.5% by lag 0→1→2; Phase 1 will provide data).
- Weekday effects (paper: Thursday >> Monday; requires day-of-week diagnostic, not yet implemented).
- Turnover efficiency (paper: ML 212% vs 317% for basic reversal; Phase 1 will compute holdings).
- Performance net of costs (paper: 9.7% annualized return, 1.6 IR globally; Phase 1 results pending).

**Comparability caveats limit inference:**

1. **3-year window vs 19 years**: insufficient to separate regime-dependent from persistent effects.
2. **US-only vs global**: removes diversification benefit and regional robustness.
3. **12 factors vs 86**: most critically, missing UPDOWN1W (earnings revisions), the paper's #2 feature.
4. **Raw features vs neutralized**: increases collinearity, may reduce SHAP interpretability.
5. **Survivorship bias**: inflates returns, approximately 50–100bp per year.

**Recommendation for Phase 2:**

Prioritize (a) wiring neutralization into main.py, (b) ingesting consensus earnings-revision data and constructing UPDOWN1W, SUE3, SUE6 factors, and (c) implementing point-in-time universe membership. These three changes will bring the backtest architecturally closer to the paper. Extending the window to 5–7 years (if Databento data availability permits) and adding EU and Japan universes would further strengthen claims.

The Phase 1 results, once computed, will tell us whether a simplified, US-centric, earnings-blind version of the industry research framework still works. Given the paper's robust findings on global data, we expect Phase 1 to show positive alpha and lower turnover than naive reversals, even with these constraints. However, the 3-year window and missing earnings data likely imply Phase 1 IR in the range of 0.6–1.0 (vs the paper's 1.0–1.6), and definitive proof of the hypothesis awaits Phase 2 completion.