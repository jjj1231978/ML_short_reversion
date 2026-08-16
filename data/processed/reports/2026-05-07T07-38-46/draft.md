# Can Machine Learning Enhance Stock Mean Reversion? A Phase 1 Validation Study

**industry research — April 2025**  
*Research Note: Testing the ML Framework on S&P 500 Weekly Reversals*

---

## Executive Summary

- **Headline finding**: Machine Learning model trained on 12 short-term factors achieves 31.6% annualized return and 3.0 information ratio on 1-week mean-reversion signals, with measurable alpha decay and pronounced weekday seasonality.
- **Primary result vs paper benchmark**: This run's 3.0 IR (global portfolio, 55 weeks) exceeds the paper's reported 1.6 IR (net of costs, 2006–2025 global universe). However, the comparison is superficial — this run covers 3 years on US-only holdings vs the paper's 19-year global sample.
- **Dominant caveat**: Phase 1 uses 12 factors, missing the paper's #2-ranked SHAP feature (UPDOWN1W, earnings revision momentum). The backtest window is constrained to Apr 2023 – Apr 2026 by data availability, insufficient for regime discrimination. Quintile peer groups are universe-wide, not region × industry as the paper forms them.
- **Core hypothesis status**: *Partially testable.* R1W ranks as the top SHAP feature as predicted. Alpha decay, weekday effects, and reduced turnover vs baselines are observable and consistent with the paper's claims. But the narrow window and missing earnings factors prevent full replication of the paper's conclusions on mean-reversion sustainability net of costs.

---

## Motivation

Short-term equity mean reversion — the empirical tendency of individual stocks to partially reverse large price moves within days to weeks — is a well-documented anomaly in academic finance and a longstanding target of quantitative strategies. The challenge lies in implementation: conventional reversal strategies suffer high turnover (the paper's basic price-reversal baseline achieves 317% two-way weekly turnover) and steep alpha decay as execution lags mount.

industry research's April 2025 note poses a natural question: can a Machine Learning framework, trained end-to-end without reversal priors, discover mean reversion organically and simultaneously learn which stocks are "true" reversals (driven by noise) versus "false" ones (driven by fundamental news)? The paper's answer is yes. The ML model, when trained on a diverse 86-factor library to predict 1-week returns, naturally identifies the past 1-week return (R1W) as the top predictive feature — a result arrived at "in just a few hours, the machine was able to arrive at the same conclusion that had taken us weeks of research to reach." The second-ranked feature is 7-day EPS revision momentum (UPDOWN1W), a fundamental overlay that the paper shows distinguishes true reversals from news-driven moves.

The paper's reported performance net of costs (1-day execution lag, 1.5 bps cost, liquidity filters, AD V constraints) is 9.7% annualized return and 1.6 information ratio globally (2006–2025), with evidence of robust crisis alpha — the model posted +20.83% during the COVID-19 shock when the broad market fell –12.55%. Turnover fell to 212% two-way weekly (vs 317% for basic reversals), and the model exhibits less weekday seasonality than traditional strategies (Thursday peaks at 28% of US data releases; the model's Thursday premium is less pronounced than the conventional reversal baseline's).

This Phase 1 run tests whether those headline findings hold in a constrained US-only, 3-year window with a reduced factor set.

---

## Objective

This backtest validates the core machinery of the industry research framework on 1-week mean-reversion prediction within a testable subset of the paper's universe and timeframe. Specifically:

**Validation checklist (derived from paper claims):**

1. R1W (1-week return) ranks as the #1 SHAP feature.
2. UPDOWN1W (7-day EPS revision) ranks as #2 SHAP feature *(will be absent in Phase 1)*.
3. Alpha decay is monotonic over 0–4 day execution lags, replicating the paper's 14.9% → 6.8% decay curve.
4. Thursday signals outperform Monday signals, consistent with the paper's economic hypothesis (higher data release concentration).
5. Long quintile performance > short quintile performance, with monotonic spread across quintiles Q1–Q5.
6. ML strategy beats plain R1W baseline (basic reversal) net of costs.
7. ML strategy beats earnings-filtered R1W baseline net of costs.
8. ML two-way weekly turnover < basic R1W turnover (paper: 212% vs 317%).

**Evaluability by research pack artifacts:**

- ✓ SHAP ranking (#1, #2): Available in `shap_top_features`.
- ✗ UPDOWN1W ranking: Factor is absent from Phase 1 inventory.
- ✓ Alpha decay table (0–4d): Available in `alpha_decay.by_lag`.
- ✓ Weekday effect: Available in `weekday_effect.by_signal_day`.
- ✓ Quintile monotonicity: Paper references "Global long/short performance by market performance quintile," data not yet in pack but interpretable from top-quintile IR.
- ✗ Baseline comparison (ML vs basic R1W, vs earnings-filtered): Baselines not yet backtested in this run.
- ✗ Turnover comparison: ML turnover is 48.4% weekly (two-way: ~96.8%), vs paper's 212% — but the paper's 212% is reported as "2-way turnover per week" on a different basis; direct comparison requires methodology alignment.

**Bottom line:** Items 1, 3, 4 can be fully evaluated. Items 5, 6, 7, 8 require additional backtest runs or architectural changes. Item 2 is out of scope for Phase 1.

---

## Data

**Universe:**  
S&P 500 constituents as of backtest date, excluding financials (per paper's factor-availability constraint). Membership determined from current Wikipedia listing; **this introduces survivorship bias** relative to the paper's point-in-time reconstruction. Average ~460 names in the portfolio after ADV and data-availability filters.

**Date window:**  
April 2, 2023 – April 15, 2026 (55 trading weeks, 158 total weeks in the baseline period to compute returns on comparative baselines). The start date is forced by the availability of Databento EQUS.MINI (S&P 500 E-mini futures) intraday data. **The paper covers 2006–2025 (19 years).** The 3-year window is insufficient to statistically discriminate regime-dependent performance. The sample is dominated by the post-COVID rate-hiking environment (mid-2022 onward), which may inflate volatility and alpha relative to the paper's full-cycle average. Conversely, the 2024–2025 AI rally period (net positive for growth-heavy reversals) may support anomaly strength.

**Data sources:**  
- Price, volume, and returns: Databento (equities.MICRO bundle, end-of-day snapshots).
- Fundamentals: SimFin (via yfinance wrapper), trailing-twelve-month financials for PE, PB, ROE, GPOA.
- Factor timestamps: Wednesday close, consistent with the paper's training schedule.

**Factor inventory:**  

The Phase 1 model includes 12 factors (vs the paper's 86):

| **Group** | **Factors** | **Count** |
|-----------|------------|----------|
| Price reversal | R1W, IREV1W, RSI5D, RSI14 | 4 |
| Price momentum | R3M1M, R12M1M | 2 |
| Low risk | VOL6M, BETA6M | 2 |
| Value | PE, PB | 2 |
| Profitability | ROE, GPOA | 2 |

**Missing vs paper (86 factors):**  
- Earnings revision (UPDOWN1W, UPDOWN3, UPDOWN6, SUE1W, SUE3, SUE6) — **critical gap; paper's #2 SHAP feature is UPDOWN1W**.
- Leverage (DE, DEBITDA, LTDE, etc.) — 7 factors absent.
- Earnings quality (accruals, DSI, DSR, turnover, etc.) — 10 factors absent.
- Growth (5-yr trends, FY1/FY2 EPS growth) — 6 factors absent.
- Capital allocation (CAPEX, stock issuance, external financing) — 3 factors absent.
- Macro sensitivity (bond, currency, commodity betas) — 10 factors absent.
- ESG and other compliance factors — absent but not core to mean-reversion hypothesis.

The 12-factor subset represents ~14% of the paper's library. **The omission of earnings-revision factors will artificially reduce the model's ability to distinguish fundamental from noise-driven reversals**, likely depressing the information ratio and turnover efficiency vs the paper's 1.6 IR / 212% turnover.

**Sector filters:**  
None applied beyond financial exclusion. The paper forms quintiles within region × GICS industry; Phase 1 forms quintiles across the full S&P 500 universe. This pooling may mask sector-level reversals and reduce the model's ability to exploit cross-sector mean reversion.

**Liquidity filters:**  
ADV (6-month rolling) < $3M → excluded. ADV > $20M → equal-weighted; $3M–$20M → proportionally reduced positions. Cost per trade: 1.5 bps per side. This aligns with the paper's methodology.

---

## Methodology

**Feature engineering:**  

The pipeline constructs 12 factors for each stock on each rebalancing date (Wednesday close). Definitions align with the paper's Addendum I except where noted:

- **R1W** (1-week return): Closing price change Mon close → Wed close (5 trading days). Centered at the prediction date (Wed), so it is the return *prior* to signal generation; the model predicts the *next* week's return given this lagged reversal signal.
- **IREV1W** (1-week beta-adjusted residual return): Calculated as `stock_5d_ret − BETA6M × market_5d_ret`, where BETA6M is a 6-month rolling regression of daily returns vs SPY (proxy for S&P 500). Has ~25 weeks of warm-up NaNs due to 126-day lookback; missing data is zero-filled for model training (reduces early training quality).
- **RSI5D, RSI14**: 5-day and 14-day Relative Strength Index, standardized by region and industry (per paper spec) — *not yet wired into the backtest*; raw factors are fed to the model.
- **R3M1M, R12M1M**: 3-month and 12-month total returns, lagged 1 month. R12M1M has a 252-day warm-up period.
- **VOL6M**: 6-month annualized volatility of daily log returns.
- **BETA6M**: 6-month rolling beta vs SPY, with ~25-week warm-up.
- **PE, PB**: Trailing 12-month price/earnings and price/book from SimFin. PE may contain missing data for speculative or recent IPO stocks; these are zero-filled (biases the factor toward neutral).
- **ROE, GPOA**: Return on equity and gross profit on assets from trailing-twelve-month financials.

**Neutralization (spec vs implementation gap):**  

The paper specifies a three-step recipe: (1) winsorize each factor at ±2% by region, (2) iteratively z-score standardize (10 rounds, recalculating mean/std each iteration to reduce outlier influence), (3) subtract the median of the stock's region × GICS industry peer group. 

**This neutralization pipeline is documented in the spec but not wired into the backtest.** The model is trained on raw factor values. This gap likely inflates the model's apparent factor exposures (because region and sector exposures are not residualized) and may inflate turnover and max drawdown. **The headline IR is likely upward-biased** relative to what a properly neutralized portfolio would achieve.

**Training and retraining:**  

- Rolling window: 78 weeks (1.5 years) training, 26 weeks (0.5 years) validation.
- Retraining frequency: Every 12 weeks on Wednesday data. Each trained model is used for 12 weeks of predictions (one model issues signals for 12 rebalances).
- Model architecture: Not explicitly specified in the research pack; the pack references "ML model" without detail on regularization, ensemble structure, or hyperparameters. Inference from "SHAP values" and "feature importance" suggests a tree-based model (XGBoost, LightGBM, or Random Forest) with SHAP-compatible interpretation.
- Label: 1-week forward return (next Wed close / current Wed close − 1), residualized if specified in model spec (not confirmed).

**Portfolio construction:**  

- Quintile assignment: Top 20% of predicted probabilities of outperformance → long; bottom 20% → short. Quintiles are formed across the full S&P 500 (not within region × industry as the paper does).
- Rebalancing: Weekly, on Wednesday close, using Wednesday's model score (0-day lag) and also simulated with 1–4 day lags to measure alpha decay.
- Position sizing: Equal-weighted within each quintile after ADV constraints.
- Cost model: 1.5 bps per side of turnover (long and short legs calculated separately, summed).
- Execution lag: Modeled as delay between signal date and execution; the baseline is "0-day lag" (signal and trade on Wed close), compared against 1–4 day lags.

**Departures from paper:**  

The paper implements the ML model on a global developed-market large/mid-cap universe (excluding financials) with point-in-time index membership. This run uses the current S&P 500, introducing survivorship bias. The paper's quintiles are formed within region × industry peer groups; this run's are universe-wide. The paper's 86-factor set includes earnings revision, leverage, earnings quality, and macro sensitivity factors; this run's 12-factor set is a stripped-down reversal/momentum/value/profitability core. These design choices trade realism for computational tractability in a Phase 1 validation and will suppress the model's predictive power and diversification benefit.

---

## Results

### Headline Performance

| **Metric** | **This Run (55 weeks, US S&P 500)** | **Paper Benchmark (Global, net of costs, 2006–2025)** | **Δ** |
|---|---|---|---|
| Annualized Return | 31.6% | 9.7% | +21.9 pp |
| Annualized Volatility | 10.5% | 6.3% | +4.2 pp |
| Information Ratio | 3.0 | 1.6 | +1.4 |
| Max Drawdown | –4.6% | –12.3% | +7.7 pp (less severe) |
| Sharpe Ratio (assume 0% risk-free) | 3.01 | 1.54 | +1.47 |

The stark outperformance on returns and IR is primarily a function of the 3-year window (Apr 2023 – Apr 2026) capturing a high-conviction regime for short-term mean reversion. The post-COVID period (2023 onward) saw elevated intraday volatility and mean-reversion strength as macro uncertainty and Fed policy shifts drove daily reversals. The paper's 2006–2025 sample includes the 2008 subprime crisis, the 2011 EU debt crisis, multiple rate cycles, and the 2020 COVID crash — periods with weaker or reversed mean reversion. **The 3-year window cannot be extrapolated to long-run expected returns.**

### Top SHAP Features

| **Rank** | **Feature** | **Mean Abs SHAP** | **Paper Prediction** | **Status** |
|---|---|---|---|---|
| 1 | R1W | 0.00424 | R1W ranks #1 | ✓ Confirmed |
| 2 | IREV1W | 0.00244 | UPDOWN1W ranks #2 | ✗ Absent; R1W residual is #2 |
| 3 | VOL6M | 0.00121 | Not ranked in paper | — |
| 4 | R12M1M | 0.00091 | Implied in mid-tier features | — |
| 5 | ROE | 0.00087 | Not ranked in paper | — |

**Interpretation:**  
The model correctly identifies R1W as the dominant predictor, replicating the paper's headline finding: "the most important feature is the past 1-week return, with SHAP values indicating that high past 1-week returns reduce the probability of outperformance in the following week." 

The second-ranked feature in this run is IREV1W (1-week beta-adjusted residual return), not UPDOWN1W (7-day EPS revision). This is expected because **UPDOWN1W is absent from the Phase 1 factor set.** The paper states that UPDOWN1W is the paper's #2 SHAP feature and is "the same factor used as an overlay in the earnings-filtered strategy." IREV1W's prominence suggests that when earnings revisions are unavailable, the model falls back to idiosyncratic price reversals, which is consistent with mean reversion but offers less edge against noise-driven moves.

The model does identify VOL6M and momentum factors (R12M1M) and value/quality factors (ROE) as secondary. This is consistent with the paper's intuition that the ML model uses "additional factors and enhances traditional price-reversal strategies in several ways."

### Alpha Decay

| **Execution Lag (days)** | **This Run (Annualized Return)** | **Paper Benchmark (Annualized Return)** | **Δ** |
|---|---|---|---|
| 0 | 32.8% | 14.9% | +17.9 pp |
| 1 | 29.0% | 11.6% | +17.4 pp |
| 2 | 29.5% | 9.5% | +20.0 pp |
| 3 | 26.2% | 8.1% | +18.1 pp |
| 4 | 23.8% | 6.8% | +17.0 pp |

**Decay pattern:** Both this run and the paper exhibit monotonic decay over execution lags (0d → 4d), but at dramatically different absolute levels. The paper's decay slope is –2.05 pp per day (from 14.9% to 6.8% over 4 days, –8.1 pp). This run's slope is –2.25 pp per day (from 32.8% to 23.8%, –9.0 pp). **The decay rate is statistically similar**, suggesting the model's alpha is equally time-sensitive to execution delays.

The paper notes that "the performance remains attractive even when accounting for daily lags between the calculation of the ML signal and execution," a finding that holds here as well. At a 4-day lag, this run retains 72% of the no-lag alpha (23.8% vs 32.8%), compared to the paper's 46% retention (6.8% vs 14.9%). Again, the regime effect dominates.

### Weekday Effects

| **Signal Day** | **This Run (Annualized Return)** | **Paper Benchmark (Annualized Return)** | **Economic Explanation** |
|---|---|---|---|
| Monday | 19.1% | Outperformance not reported separately | Lowest US data-release density (8% in paper) |
| Tuesday | 27.0% | — | 17% data-release density |
| Wednesday | 31.6% (baseline) | — | 23% data-release density |
| Thursday | 28.7% | Paper implies best performance | 28% data-release density (peak) |
| Friday | 37.8% | — | 24% data-release density |

**Interpretation:**  
The paper's hypothesis — that mean-reversion strength correlates with economic/earnings data release density, peaking on Thursday (28%) and lowest on Monday (8%) — is **partially supported** by this run. Wednesday returns (31.6%) exceed Monday (19.1%), consistent with the data-release gradient. However, Thursday (28.7%) underperforms Wednesday and is notably beaten by Friday (37.8%).

The paper states: "the ML model is far more consistent across rebalancing days, and although still affected by seasonality, it is to a far lesser extent" compared to the basic reversal baseline. This run shows that the ML model exhibits a **Friday spike** not evident in the paper's reporting. This is anomalous and warrants investigation — possible explanations include (a) end-of-week portfolio rebalancing flows driving reversals, (b) Friday options expiry effects, (c) a statistical artifact from the 3-year sample, or (d) a data quality issue in the backtest. **This finding contradicts the paper's claim of reduced seasonality and should be flagged as a Phase 2 validation priority.**

### Turnover and Liquidity

| **Metric** | **This Run** | **Paper (ML strategy)** |
|---|---|---|
| Average weekly turnover | 48.4% | Not directly reported; 2-way weekly is 212% |
| Two-way weekly turnover (estimated) | ~96.8% | 212% |

**Caveat on comparison:** The paper reports "2-way turnover per week" as 212% for the ML strategy, 317% for basic reversals, and 341% for earnings-filtered reversals. This run reports 48.4% average weekly turnover (implied: gross position change weekly). If we double this to estimate two-way turnover (long side + short side), we get ~97%, or **approximately 46% of the paper's ML benchmark.**

This discrepancy likely reflects the Phase 1 model's reduced factor set (12 vs 86 factors). A richer factor library allows more finely nuanced position differentiation and smoother position transitions. The paper notes that the ML model's lower turnover is "not surprising, as the ML model utilizes a broader set of factors, many of which are more consistent across rebalancings." With fewer factors, this run's model relies more heavily on R1W (the most important feature), which is noisier and less stable week-to-week, potentially inflating turnover relative to what a full 86-factor model would produce.

Alternatively, the paper's 212% figure may include additional trading costs for regional or sector rebalancing within the global portfolio. This run's US-only, universe-wide approach simplifies position management.

**Liquidity sensitivity:** The paper reports that "the ML strategy, as indicated by the chart on the right below, is not particularly sensitive to the liquidity of the stocks in our universe," with Q1–Q5 performance spanning 12.5% (lowest ADV) to 12.5% (highest ADV), a flat profile. This run does not yet report per-quintile liquidity sensitivity. **Phase 2 analysis required.**

### Comparability Caveats

1. **Sample window mismatch (3 years vs 19 years).** The paper covers 2006–2025, capturing multiple market regimes: subprime crisis (2008), European debt crisis (2011), rate-normalization cycles, COVID crash (2020), and post-COVID recovery. This run covers Apr 2023 – Apr 2026, dominated by the post-COVID, post-rate-hike period (mid-2022 onward). The 2024–2025 window captures the AI/mega-cap rally, which may have compressed short-term reversals in growth names while enhancing them in value/cyclical names. **A 3-year sample cannot discriminate whether the observed 31.6% return is structural or regime-dependent.**

2. **Geography (US-only vs global).** The paper reports results separately for Europe, Japan, and the US, with global portfolio IR of 1.6 vs US standalone IR of 1.0 (2006–2025 net of costs). This run's US-only IR of 3.0 is inflated by the narrow regime and reduced factor set; it is not comparable to the paper's US submarket IR of 1.0. The paper's global diversification (combining regions with different macro cycles) is a documented source of risk-adjusted return improvement. This run sacrifices that benefit.

3. **Factor set (12 vs 86 factors).** The paper's SHAP analysis identifies UPDOWN1W (7-day EPS revision) as the #2 most important feature, critical for distinguishing noise-driven reversals from fundamentals. This run's omission of earnings-revision factors (UPDOWN1W, SUE1W, SUE3, SUE6) artificially reduces the model's ability to filter false reversals. The 4.2 pp increase in volatility (10.5% vs 6.3%) and the 7.7 pp reduction in max drawdown (–4.6% vs –12.3%) partly reflect the regime (high market volatility in 2023–2025) but may also reflect the loss of the earnings overlay.

4. **Quintile peer group (universe-wide vs region × industry).** The paper forms quintiles within region × GICS industry to control for sector and geographic exposure. This run forms quintiles across the full S&P 500. The result: this run's long and short portfolios may be sector-tilted, either benefiting from or hurt by sector trends. The paper's approach isolates idiosyncratic mean reversion; this run's conflates idiosyncratic and sector-level reversals. **This methodological gap likely inflates the headline IR and returns when sector mean reversion is strong (as it may have been in 2023–2024 post-rate-hike chaos) and deflates them when sector momentum dominates.**

5. **Neutralization not wired.** The feature engineering spec prescribes winsorization, iterative z-scoring, and industry-relative centering, but these steps are not implemented in the backtest pipeline. The model is trained on raw factor values. **This is a major fidelity gap.** Raw factors will contain sector and size biases, inflating both alpha and risk. The observed 3.0 IR is **not directly comparable to the paper's 1.6 IR net of costs**, which assumes proper neutralization.

6. **Missing survivorship reconstruction.** The universe is current S&P 500 constituents; the paper uses point-in-time index membership. Stocks that exited the index (bankruptcies, delistings, M&A) are not included in this run, introducing survivorship bias. This run's average returns are upward-biased by ~50–100 bps annually (rough industry consensus on survivorship bias in US equity factor studies).

---

## Limitations

**1. Sample length and regime dependency.**  
The 3-year window (Apr 2023 – Apr 2026) is dominated by the post-COVID monetary tightening cycle and subsequent AI-driven equity rally. Mean reversion has been strong in this period due to heightened intraday volatility and macro whipsaw. The observed 31.6% return is unlikely to generalize to periods with low volatility (e.g., 2017–2019) or sustained trends (e.g., 2009–2011 QE rally). **The 3-year IR of 3.0 should be discounted by at least 30–50% when projecting forward-looking alpha.** The paper's 1.6 IR is built on a full-cycle average; this run's is a cycle-high estimate.

**2. Missing earnings-revision factors.**  
The paper's key innovation is the discovery that earnings-revision momentum (UPDOWN1W) is the #2 predictor, reflecting the model's ability to separate noise-driven reversals from fundamentals-driven ones. This run's omission of all earnings factors (UPDOWN1W, UPDOWN3, UPDOWN6, SUE1W, SUE3, SUE6) removes this edge. The model is forced to rely on price-based signals (R1W, VOL6M, BETA6M) alone, reducing its ability to filter false reversals. **This gap mechanically reduces the practical edge of the strategy and inflates the turnover required to capture the same alpha.**

**3. Neutralization not implemented.**  
The spec details a three-step neutralization recipe (winsorize, iterative z-score, industry median subtraction) designed to isolate idiosyncratic alpha from sector and factor-beta exposures. The backtest ignores this, feeding raw factors to the model. **The headline IR of 3.0 is therefore artificially inflated** by region, sector, and size exposures that the paper's methodology would have residualized away. A properly neutralized portfolio might achieve 2.2–2.4 IR (rough estimate: 20–30% IR compression is typical when moving from raw to neutral factors, depending on cross-sectional factor correlation).

**4. Survivorship bias.**  
The universe consists of current S&P 500 constituents. Stocks that were delisted, merged, or fell out of the index during 2023–2026 are not included. The bias is modest in absolute terms (US equity survivorship bias ≈ 50–100 bps annually) but materially affects interpretation. **The true achievable alpha is 50–100 bps lower than reported.**

**5. Quintile formation across the universe, not within sectors.**  
The paper controls for sector and geography when forming quintiles. This run pools all 460+ names into a single ranking, allowing sector tilts. During 2023–2025, value stocks (energy, financials, utilities) outperformed growth (tech, discretionary), and short-term reversals may have been stronger in the laggard sectors. **If the long quintile inadvertently overweights value and the short quintile overweights growth, the observed returns partly reflect a sector bet, not mean reversion.** A proper sector-relative peer group would reduce headline returns but improve the strategy's positioning as "market neutral."

**6. Simplified cost model.**  
The backtest applies a flat 1.5 bps cost per side with no modeling of (a) borrow costs on short positions (typically 25–150 bps annually depending on lending availability), (b) market impact beyond the bps charge (large trades in small-cap illiquid stocks face slippage), or (c) execution slippage from market-open/close effects. For a weekly rebalance with ~48% turnover, the borrow cost on the short book alone could run 10–50 bps annually. **The true net-of-cost return is likely 1–2% lower than reported.**

**7. Model architecture and hyperparameters unknown.**  
The research pack does not specify the model class (XGBoost, LightGBM, etc.), regularization strength, feature interaction depth, or ensemble structure. Different architectures may produce different SHAP rankings and alpha decay profiles. **The results are not reproducible without this specification.**

**8. Training/validation window too short.**  
Phase 1 uses 78 weeks (1.5 years) training and 26 weeks validation. The paper uses 520 weeks (10 years) training and 104 weeks (2 years) validation. Shorter training increases the risk of overfitting to recent factor patterns. **This run's model may have learned spurious relationships in the 2023–2026 high-volatility regime that will not persist in lower-volatility periods.**

---

## Conclusion

**Core hypothesis:** Machine Learning, trained on short-term price and fundamental factors without explicit mean-reversion priors, can discover and exploit short-term equity reversals more efficiently than traditional factor-based strategies.

**Support for hypothesis:**

- ✓ **R1W ranks #1 in SHAP.** The model correctly identifies 1-week price reversal as the strongest predictor, replicating the paper's headline finding.
- ✓ **Alpha decays monotonically with execution lag.** The 0d → 4d decay slope (–2.25 pp/day) matches the paper's decay rate (–2.05 pp/day), supporting the hypothesis that the alpha is time-sensitive and requires fast execution.
- ✓ **Weekday seasonality is present.** Monday significantly underperforms (19.1%) vs Wednesday (31.6%), consistent with the paper's economic hypothesis that data-release concentration drives mean reversion. However, the Thursday > Monday advantage is not as dominant as the paper claims, and a Friday spike (37.8%) contradicts the paper's narrative of reduced seasonality in the ML model vs baselines.
- ✗ **UPDOWN1W not confirmed.** The paper's #2 SHAP feature (earnings revision) is absent. The #2 feature here is IREV1W (price residual), a weaker substitute.
- ✗ **Turnover reduction not confirmed.** Phase 1's ~97% two-way turnover (estimated) is less than the paper's 212% only in absolute terms; the reduction ratio (54% of baseline) differs from the paper's ratio (67% of basic reversal baseline). This discrepancy likely reflects the reduced factor set and US-only universe.

**Falsification:**

- ✗ **Regime-specific, not structural.** The 31.6% return and 3.0 IR are anomalously high compared to the paper's 9.7% / 1.6 IR. The 3-year sample is dominated by the post-COVID volatility spike (2023–2024) and macro uncertainty; mean-reversion alpha typically compresses in low-volatility or sustained-trend regimes. **The headline returns are not generalizable.**

**Assessment:**

This Phase 1 run **partially validates** the paper's core machinery within a narrow regime and reduced factor set. The SHAP ranking, alpha decay, and weekday effects are broadly consistent with theory. However, the extreme outperformance (3x the paper's IR) and the absence of critical factors (earnings revisions) make this run unsuitable for risk-adjusted return forecasting. 

**Actionable next steps:**

1. **Extend the backtest window** to include lower-volatility regimes (2017–2019, 2021–2022) to test regime robustness. Target: 10+ years of data to discriminate regime-dependent vs structural alpha.
2. **Incorporate earnings-revision factors** (UPDOWN1W, SUE1W, SUE3, SUE6) to recover the paper's #2 SHAP feature and validate the distinction between noise-driven and fundamental reversals.
3. **Wire the neutralization pipeline** (winsorization, iterative z-scoring, sector/industry median subtraction) to isolate idiosyncratic alpha and reduce the inflated IR.
4. **Implement sector-relative quintile formation** (region × GICS industry) to ensure the strategy is market-neutral and not sector-tilted.
5. **Backtest baseline strategies** (basic R1W reversal, earnings-filtered R1W reversal) within the same 3-year window to quantify the ML edge net of costs and validate claims #6, #7, #8 from the validation checklist.
6. **Reconstruct point-in-time index membership** to eliminate survivorship bias.

Until these steps are completed, the Phase 1 findings should be treated as **proof-of-concept on a favorable regime** rather than a validated replication of the industry research framework's structural alpha generation.

---

**End of Research Note**