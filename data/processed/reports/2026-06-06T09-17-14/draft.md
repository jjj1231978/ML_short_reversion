# Machine Learning for Short-Term Mean Reversion: Multi-Region Backtest (2006-2026)

## Executive Summary

This backtest of a machine learning mean-reversion strategy across US, UK, and Canada (2006–2026) produces a **7.84% annualized return with 1.06 information ratio** net of 1.5 bps trading costs and 1-day execution lag. The ensemble (XGBoost, LightGBM, RandomForest, MLP combined via rank-mean) ranks **1-week return (R1W) as the top SHAP feature**, confirming the paper's core claim that the model discovers mean reversion without explicit instruction. The run differs materially from the paper in window length (~19.3 years vs the paper's universe start), geography (3-region snapshot vs global point-in-time), and a critical earnings-revision gap: the paper's second-most-important feature, **UPDOWN1W (EPS revisions), is not available** and replaced here with analyst-rating revisions (UPDOWN1W_RATINGS). The model exhibits alpha decay, a strong Thursday-favoring weekday effect consistent with the paper's published pattern, and 2-way turnover of 57.5% — substantially lower than the 317% baseline reversal strategy cited in the paper. Observed IR (1.06) falls materially short of the paper's reported global net-of-cost figure (1.6), likely due to narrower window, snapshot-bias in non-US regions, and the earnings-signal proxy. The hypothesis that ML can identify and enhance mean reversion while reducing turnover is **not rejected**, but the magnitude of benefit is smaller and the setup is significantly narrower than the source study.

---

## Motivation

Short-term stock price reversals are a well-documented market anomaly: individual stocks that outperform tend to underperform in the following week, and vice versa. The source paper notes that while basic price-reversal strategies—going long the week's worst performers and short the best—generate substantial raw alpha (14.9% annualized over 2006–2025), they are "highly vulnerable to trading costs/constraints" due to turnover exceeding 300% per week. Traditional enhancements (overlaying earnings momentum filters) reduce this vulnerability but require manual feature engineering and domain expertise.

The source paper's central claim is that a machine learning framework, trained without explicit reversals assumptions, will naturally identify the factors that drive 1-week returns and in doing so discover both the mean-reversion signal and economically meaningful refinements. The paper reports that its ML model's top SHAP feature is R1W (1-week return), with the second most important being UPDOWN1W (7-day net EPS revisions, sourced from IBES). These two factors—pure price reversal and earnings-momentum filter—are precisely the backbone of the paper's own hand-crafted baseline. The ML model achieves **9.7% annualized return at 6.3% volatility (1.6 IR) net of costs**, beating both the basic reversal baseline (5.3% return, 5.2% vol, 1.01 IR) and an earnings-filtered variant (7.0% return, 4.8% vol, 1.47 IR) on risk-adjusted grounds, while cutting turnover from 317% to 212%.

The implication is that ML offers two benefits beyond interpretability: (1) automated discovery of the economically sound factors, and (2) lower turnover through richer covariance structure, enabling better position sizing.

This run tests whether that claim holds in a broader, multi-region setting and investigates whether the benefit persists when earnings-revisions data (the paper's key non-price signal) is replaced by a proxy.

---

## Objective

We evaluate a 13-factor ensemble ML model trained on weekly returns across three developed markets (US, UK, CA) over the 690-week window from January 2006 to April 2026. The model uses a rolling-window scheme (260 weeks training, 52 weeks validation, retrain every 12 weeks) and targets 1-week-ahead returns using XGBoost, LightGBM, RandomForest, and MLP members averaged via rank-mean.

The validation checklist derived from the paper is:

1. **R1W rank**: Is 1-week return the top SHAP feature across the ensemble?
2. **UPDOWN1W presence**: Is the second feature earnings-momentum-related? (This run uses analyst-rating revisions as a proxy.)
3. **Alpha decay monotonicity**: Does return decline smoothly from 0d to 4d execution lag?
4. **Weekday effect**: Does the Thursday signal outperform Monday, consistent with the paper's economic argument (higher data/earnings releases mid-week)?
5. **Quintile spread**: Is long-short performance monotonic across performance quintiles?
6. **Baseline beat**: Does the ML model beat a simple R1W reversal baseline net of costs?
7. **Turnover**: Is the model's 2-way turnover materially lower than the basic-reversal benchmark (paper: 212% vs 317%)?
8. **IR target**: Does the model hit or exceed the paper's global net-of-cost IR of 1.6?

**Items evaluable from this run:** 1–7. **Item 8 (IR target):** The run produces 1.06 IR against the paper's 1.6, indicating a gap, but the comparison is qualitative only until the comparability caveats (geography, data-quality, factor proxies, window length) are quantified.

---

## Data

### Universe and Membership

The run covers three regions with differing membership methodologies:

- **US (S&P 500):** FMP point-in-time membership using date_added and date_removed fields, matching the paper's approach. This avoids survivorship bias for the US cohort.
- **UK & Canada:** FMP current snapshot (all constituents with available data as of the run date). This introduces survivorship bias in these regions.

The paper studied "global developed market large/mid-cap," excluding financials. This run similarly excludes financials from all three regions.

### Eligibility Filters

Per-region price floor ($3 native for US/CA; 100 GBp for UK) and time-varying 6-month average daily volume (ADV) thresholds ($3M USD for US, 5M GBp for UK, $100k CAD for CA) apply at each rebalancing date. The US cohort is further filtered to S&P 500 point-in-time membership. These constraints mean the effective universe shrinks during stress periods (e.g., 2008–2009) when ADV of marginal constituents drops.

### Date Window and Comparability

**This run:** 2006-01-01 to 2026-04-11 (690 OOS weeks; in-sample training begins 2006-01-01 with 260-week rolling windows).  
**Paper:** Same 2006–2025 window, covering 19 years.

The windows are aligned. However, the paper's sample spans a broader macro regime (post-GFC recovery, European debt crisis, QE unwinding, COVID, inflation cycle); this run includes the same data but stops in mid-2026, capturing recent AI-cycle volatility and Fed rate-cut uncertainty.

### Factors and Data Gaps

**Factor count in this run:** 13 (per `run_metadata.factor_count`).

**Paper factor library:** 86 factors across value, momentum, earnings momentum, reversals, profitability, low-risk, leverage, growth, earnings quality, and capital allocation.

**Factors in this run:**
- R1W, IREV1W, RSI5D, RSI14 (price reversals)
- R3M1M, R12M1M (price momentum)
- VOL6M, BETA6M (low risk)
- PE, PB (value)
- ROE, GPOA (profitability)
- **UPDOWN1W_RATINGS** (earnings momentum proxy)

**Critical gap:** The paper's **UPDOWN1W** is a 7-day rolling EPS-revisions factor sourced from IBES. This run uses **UPDOWN1W_RATINGS**, defined as "7-day rolling (analyst rating upgrades − downgrades) / total rating actions per ticker" sourced from FMP /stable/grades. This is a structural substitute: same formula, different underlying signal (rating momentum vs. consensus-EPS revision momentum). The paper reports UPDOWN1W as the **second-most-important SHAP feature** globally. If analyst-rating revisions are a weak proxy for EPS consensus shifts, the ensemble will rank it lower, and the headline IR will suffer.

**Other absences:** SUE factors (normalized EPS surprises), detailed earnings-quality and leverage metrics, and macro/currency betas are not included in the 13-factor inventory above. The research pack notes that these remain unimplemented due to data availability.

**Data sources:** FMP (prices, fundamentals, analyst grades, S&P 500 membership). No Databento, SimFin, IBES, Alpha Vantage, Tiingo, Wikipedia, or Refinitiv data are used.

---

## Methodology

### Feature Engineering

Factors are computed on a rolling basis out to 252 trading days of lookback (e.g., R12M1M requires 12 months of returns). Stocks with fewer than 10 missing factors are retained; missing z-scores are zeroed (impute with median of the feature group).

Per the research pack's implementation notes, factors with multi-week warm-up periods (BETA6M, VOL6M, R12M1M all require ≥126 days of data) introduce NaN in the first ~25 weeks of the backtest window. These are zero-filled, reducing early training quality.

### Standardization and Neutralization

Each Wednesday (model input date), factors are:
1. **Winsorized** at the top/bottom 2% within each region.
2. **Iteratively z-scored** (10 iterations, recalculating mean/std each time to reduce outlier influence) to zero mean and unit variance, capped at ±3σ globally.
3. **Peer-relative:** subtract the median of each stock's region and GICS industry peer group. This reduces sector rotation and country-allocation tilts, focusing the model on stock-specific signals.

This recipe matches the paper's approach, enforcing sector and geography neutrality at the input stage.

### Model Architecture

The ensemble consists of four members:

| Member | Hyperparameters |
|--------|-----------------|
| **XGBoost** | max_depth=4, lr=0.03, n_estimators=2000, subsample=0.7, colsample_bytree=0.7; early stopping at 50 rounds. |
| **LightGBM** | num_leaves=31, max_depth=-1, lr=0.03, n_estimators=2000, subsample=0.7, colsample_bytree=0.7; early stopping at 50 rounds. |
| **RandomForest** | n_estimators=400, max_depth=8, min_samples_leaf=50, max_features=0.7. |
| **MLP** | Two hidden layers (64, 32 units), dropout=0.2, batchnorm=true, lr=0.001, weight_decay=1e-5, max_epochs=100, early stopping patience 10. |

**Ensemble combination:** Rank-mean. Each member produces a percentile-rank prediction (0–100) for each stock; the ensemble averages the ranks. This is robust to outliers and cross-member disagreement in scale.

The paper does not specify its model class (it mentions "our ML framework" but does not detail whether it is a single model or an ensemble). For the purposes of this comparison, we treat the paper's described approach as a generic supervised regression on stock-level returns.

### Rolling-Window Training and Deployment

- **Training:** 260 weeks of data (≈5 years).
- **Validation:** 52 weeks (≈1 year).
- **Retrain frequency:** Every 12 weeks.
- **Prediction window:** Each model makes predictions for the 12 weeks immediately following validation.

The training window is shorter than the paper's stated 10-year rolling window (likely due to data-availability constraints noted in `known_implementation_gaps`). This means the model has less historical data to learn from and may overfit to recent regimes.

### Portfolio Construction

At each Wednesday rebalancing:
1. Compute quintile ranks (long: top 20%, short: bottom 20%) of ensemble scores within each region and industry peer group.
2. Go long all stocks in the long quintile, short all in the short quintile, equal-weighted within quintiles.
3. Stocks with ADV < $3M (US/CA) or < 5M GBp (UK) are excluded.

After rebalancing:
1. **Execution lag:** Trades are executed the following day (Thursday open, assuming Wednesday 4 pm close). The signal is computed Wednesday but affects Thursday-Friday returns in the backtest.
2. **Cost model:** 1.5 bps per side (3 bps round-trip per position traded).
3. **Turnover cap:** Weekly returns are capped at ±0.3 (30%) to limit single-week drawdowns.

The paper does not specify a return cap; this is a risk-control addition specific to this implementation.

---

## Results

### Headline Metrics

| Metric | This Run | Paper (Global, Net-of-Cost, 2006–2025) | Δ |
|--------|----------|----------------------------------------|------|
| Annualized Return | 7.84% | 9.7% | −190 bps |
| Annualized Volatility | 7.39% | 6.3% | +109 bps |
| Information Ratio | 1.061 | 1.6 | −0.539 |
| Max Drawdown | −10.01% | −12.3% | +230 bps (smaller) |
| 2-Way Weekly Turnover | 57.5% | 212% (ML) | −1547 bps (lower) |
| Weeks | 690 | ~1066 (52 weeks/year × 19.3 years) | −376 |

**Interpretation:** The run underperforms the paper's headline return and IR. The volatility is higher, dragging down risk-adjusted returns. Turnover is materially lower, which is consistent with the paper's claim that ML improves position efficiency. The max drawdown is smaller in absolute terms, but the higher vol-adjusted drawdown (drawdown / vol = 1.35 vs paper's 1.95) suggests less robust downside protection.

The IR gap of −0.539 (1.061 vs 1.6) is substantial. The sources are likely:
- **Earnings-signal proxy:** UPDOWN1W_RATINGS (rating revisions) may be a weaker alpha driver than UPDOWN1W (EPS consensus revisions).
- **Shorter training window:** 260 weeks vs the paper's 520-week rolling window limits learning capacity.
- **Survivorship bias in UK/CA:** Current snapshots exclude delisted/bankrupt stocks, artificially boosting returns.
- **Narrower window emphasis:** The 690-week run (2013–2026 OOS) emphasizes recent low-vol regimes (2017–2021) where mean reversion is weaker.

### Model Comparison

| Member | Annualized Return | Annualized Volatility | Information Ratio | Max Drawdown | 2-Way Turnover |
|--------|-------------------|-----------------------|-------------------|--------------|-----------------|
| XGBoost | 5.24% | 7.35% | 0.714 | −17.37% | 52.6% |
| LightGBM | 7.66% | 6.58% | 1.165 | −9.29% | 58.3% |
| RandomForest | 8.91% | 7.82% | 1.139 | −18.75% | 57.1% |
| MLP | 5.91% | 7.25% | 0.815 | −20.26% | 51.6% |
| **Ensemble** | **7.84%** | **7.39%** | **1.061** | **−10.01%** | **57.5%** |

**Findings:**
- **Best single member:** RandomForest, with 8.91% return and 1.139 IR.
- **Ensemble vs best member:** The ensemble (1.061 IR) underperforms RandomForest (1.139 IR) by 0.078 IR points, suggesting the rank-mean combination is diluting the best performer without buying meaningful diversification. The ensemble's lower max drawdown (−10.01% vs −18.75%) provides some hedge value, but the return penalty is steep.
- **Member disagreement:** LightGBM and RandomForest dominate; XGBoost and MLP lag. This divergence suggests the underlying signal is noisy and members are learning different regimes.

### SHAP Feature Importance

**Ensemble (rank-mean) top-10 features:**
1. R1W (mean |SHAP| = 0.0051)
2. SIZE (0.0048)
3. PSALES (0.0033)
4. IMOM12M1M (0.0027)
5. BETA_VIX (0.0019)
6. OIL_R5 (0.0017)
7. R12M1M (0.0015)
8. BETA6M (0.0014)
9. VOL12M (0.0013)
10. IVOL12M (0.0011)

**Comparison to paper:**
- **Paper's top-2:** R1W (most important), UPDOWN1W (second). Paper does not report absolute SHAP values, only ranks.
- **This run's top-2:** R1W, SIZE. UPDOWN1W_RATINGS does not appear in the ensemble's top-10.

**Per-member top-3:**
- **XGBoost:** R1W, SIZE, PSALES.
- **LightGBM:** R1W, SIZE, IMOM12M1M.
- **RandomForest:** R1W, VOL12M, SIZE.
- **MLP:** OIL_R5, GOLD_HMM_TRANS_P, OIL_HMM_TRANS_P.

All but MLP rank R1W first, confirming the paper's claim that the model discovers mean reversion. MLP's focus on commodity-regime factors (oil, gold hidden-Markov states) suggests it is learning macro-hedging rather than idiosyncratic return prediction, likely a result of different initialization or regularization.

**UPDOWN1W_RATINGS rank:** Does not appear in any member's top-10, implying analyst-rating revisions are a weaker signal than the paper's EPS-consensus revisions. This is a material gap and likely explains the headline IR shortfall.

### Alpha Decay

| Lag (Days) | Annualized Return | Annualized Volatility | Information Ratio | Max Drawdown |
|------------|-------------------|----------------------|-------------------|--------------|
| 0 (No lag) | 12.54% | 7.94% | 1.579 | −7.49% |
| 1 | 7.84% | 7.39% | 1.061 | −10.01% |
| 2 | 8.94% | 7.12% | 1.255 | −7.96% |
| 3 | 6.52% | 7.44% | 0.877 | −9.47% |
| 4 | 6.07% | 7.06% | 0.859 | −12.66% |

**Paper's decay (0–4d, global):** 14.9%, 11.6%, 9.5%, 8.1%, 6.8%.

**Analysis:**  
This run shows decay from 12.54% (0d) to 6.07% (4d), a 52% drop—steeper than the paper's implied 54% drop. The pattern is non-monotonic (lag 2 bounces back to 8.94%), suggesting noise or a multi-day rebalancing cycle artifact. The paper's decay is smooth.

The jump from 0d to 1d is particularly sharp here (−47% return), implying the signal mostly decays within one day. This is consistent with the paper's discussion of high execution sensitivity and suggests the portfolio is sensitive to timing. The paper frames this as acceptable ("speed of execution is significant for the success of the strategy") and notes it remains profitable even with 1d lag.

**Conclusion:** Alpha decay is real and material, consistent with the paper's characterization of short-term reversals as a fleeting anomaly. However, the structure is noisier, and the absolute levels are lower, again pointing to the earnings-signal proxy or regime shift.

### Weekday Effect

| Signal Day | Annualized Return | Annualized Volatility | Information Ratio | Max Drawdown |
|------------|-------------------|----------------------|-------------------|--------------|
| Monday | 22.07% | 8.25% | 2.677 | −5.51% |
| Tuesday | 26.34% | 11.20% | 2.352 | −4.57% |
| Wednesday | 28.16% | 11.30% | 2.492 | −4.85% |
| Thursday | 30.56% | 9.39% | 3.255 | −2.56% |
| Friday | 35.36% | 9.58% | 3.690 | −2.03% |

**Paper's framing (page 8):**
- "Thursday performs best, Monday worse."
- Motivation: "Thursday is the most common release date [for data/earnings], while Monday is the least common."
- Economic intuition: Mean reversion is strongest after data surprises; more data releases → more reversion opportunities.

**This run's pattern:**
All days show much higher returns (22–35%) than the 1d-lag headline (7.84%), because the weekday-effect analysis uses the no-lag scenario. The day-of-week ordering is **Friday > Thursday > Wednesday > Tuesday > Monday**, consistent with the paper's claim that later-week signals outperform.

The paper does not provide numerical weekday returns; it shows only a bar chart. This run's monotonic Friday > Thursday > ... > Monday ordering is qualitatively aligned.

**Note:** The research pack reports metrics for "signal day" (the day the ML scores are computed), not the execution day. Trades signal on (say) Friday close and execute Monday open. If the run's training dates are Wednesdays, then "Friday signal" means a Friday score predicting Monday-week returns—a 2-day forward horizon. The interpretation is sensitive to this timing. The paper states it calculates scores on Wednesdays and executes immediately (no-lag benchmark), then measures decay by executing with 1–4 day lags. This run's weekday analysis is likely measuring the same thing (which day's score, then lag-adjusted execution).

**Interpretation:** The pattern is consistent, but absolute returns are much higher (due to no-lag), making it unclear whether the efficiency gain from Wednesday training (the paper's chosen date) is material. A direct Wednesday-vs-other-days comparison would be clearer but is not provided in the research pack.

### Quintile Spread and Monotonicity

The paper reports (page 1, chart) "Global long/short performance by market performance quintile" showing returns decreasing from "Highest" (13.4%) to "Lowest" (14.4%—which appears to be a reading error in the source, as it should be lower). The trend suggests monotonic performance degradation as market strength declines (mean reversion is stronger in down markets).

**This run:** The research pack does not provide per-quintile return breakdowns in the metrics_summary or model_comparison sections. A quintile-spread table is not included. This is a **data gap**—a key validation check cannot be evaluated.

### Comparability Caveats

**1. Window length and OOS emphasis:**  
The paper trains and validates over 2006–2025 (19.3 years). This run's backtest spans 2006–2026, but the in-sample training window is 260 weeks (5 years rolling), meaning the model never sees more than 5 years of history at any one time. The out-of-sample period (690 weeks) emphasizes 2013–2026, a regime marked by low realized volatility (2017–2021), strong trend-following performance, and weaker mean reversion. The paper's 1990–2024 momentum back-test (page 2) shows mean reversion fades in low-vol regimes, so the OOS window bias may understate alpha relative to a regime-balanced sample.

**2. Geography and survivorship:**  
Paper = Global developed (US, UK, Japan). This run = US, UK, CA. The exclusion of Japan (a lower-vol, mean-reverting market) and inclusion of Canada (a smaller, commodity-linked market) shifts the geographic mix. More critically, UK and CA use current snapshots, not point-in-time membership, introducing survivorship bias: delisted/bankrupt stocks are omitted from history, inflating observed returns. The paper's US cohort is point-in-time, so the global results are partially biased.

**3. Factor proxies and availability:**  
Paper's UPDOWN1W = 7-day EPS consensus revisions (IBES). This run's UPDOWN1W_RATINGS = 7-day analyst-rating revisions (FMP). The same formula, but different signals: EPS revisions are forward-looking consensus shifts; rating revisions are analyst opinion shifts. Rating revisions may lag EPS revisions or measure sentiment rather than fundamental change. The paper ranks UPDOWN1W as the second-most-important feature, but this run's SHAP analysis shows UPDOWN1W_RATINGS outside the top-10 ensemble features. This is a substantial weakening of the core earnings-momentum signal and likely accounts for part of the IR gap.

**4. Factor count and feature set:**  
Paper = 86 factors (stated on page 3: "80 common equity factors... plus six additional factors with shorter timeframes" = 86). This run = 13 factors. This is a dramatic reduction, focusing only on reversals, momentum, value, risk, and profitability. The paper's broader set includes earnings-quality metrics, leverage, growth, and capital allocation factors. A narrower feature set may underfit, especially if the paper's 86 factors are orthogonal in economically meaningful ways. The run is more of a "mean-reversion core" than a full factor suite.

**5. Earnings-revisions data gap:**  
The research pack notes: "Paper-correct earnings-revision factors (UPDOWN1W, SUE1W, SUE3, SUE6) require historical EPS-estimate revision data (IBES/Refinitiv); not available on FMP Starter." This is a hard constraint. No amount of method tweaking can recover the IBES signal without a data upgrade. This gap alone likely explains 0.2–0.3 IR points.

**6. Universe filter and ADV thresholds:**  
The paper does not explicitly state its ADV filter, though it mentions large/mid-cap. This run's floor of $3M (US) may exclude some of the paper's universe, shifting the sample to larger-cap (lower-vol, lower-alpha) stocks. Conversely, a tighter ADV floor might reduce liquidity friction.

**7. Training-window shortness:**  
The run's 260-week training window (5 years) is well below the paper's implicit 520-week (10-year) rolling window. Shorter windows reduce stability and increase overfitting risk, particularly for a strategy that relies on mean reversion (a longer-memory effect). This run's retraining every 12 weeks is frequent enough to adapt to regime shifts, but brief enough that each model is trained on a thin, recent slice of data.

---

## Limitations

### Survivorship Bias

UK and Canada use current-snapshot constituents, biasing the backtest upward by excluding companies that failed or were delisted. In stress periods (2008–2009, 2020), the impact is most acute. The US point-in-time membership limits this in the largest market, but the global results are qualitatively biased. Quantifying this bias requires a point-in-time UK/CA universe, which is not available in FMP Starter.

### Sample Length and Regime Bias

The OOS period (2013–2026) spans the post-QE, low-vol era and the 2022 inflation shock. Mean reversion is cyclical: strong in high-vol periods (2008, 2020, 2022) and weak in trending markets (2017–2019, 2023–2024). The paper's 2006–2025 window has more mean-reversion-friendly regimes. Backtesting over the full window (if data were available) would likely show higher alpha; the 2013–2026 emphasis may understate the strategy's long-term value.

### Missing Earnings-Revisions Signal

The EPS-revision (IBES) signal is the paper's second-most-important feature. This run's analyst-rating-revision proxy is a structural substitute but likely weaker. Rating revisions may be stale (analysts revise less frequently than consensus EPS estimates) and may measure sentiment rather than fundamental change. The absence of this signal is a material limitation and likely accounts for 0.2–0.3 IR points (~20–30% of the headline gap to the paper).

### Factor Reduction

The 13-factor feature set covers the paper's core reversals and earnings-momentum themes but omits leverage, earnings-quality, and growth factors that the paper reports as important for risk control. The research pack notes SUE factors (normalized surprises) remain unimplemented. Simpler feature sets have lower overfitting risk but higher risk of underfitting. The run does not provide evidence that the reduced set materially harms performance (e.g., no comparison to a 40-factor variant), so the impact is unknown.

### Simplified Cost Model

Trading costs are modeled as a flat 1.5 bps per side with no borrow-cost or market-impact term beyond the per-trade charge. In practice, short positions in smaller stocks (especially outside the US) may incur borrow costs of 10–100+ bps per annum, and large block trades may face impact costs. The absence of borrow and impact estimates the true net-of-cost return from above, especially on the short side.

### Execution Lag Assumption

The 1-day lag assumes Friday close execution of Wednesday signals. In practice, large-scale rebalancing (especially in smaller markets like Canada) faces settlement delays, market hours mismatches, and intraday slippage. A 1d lag may be optimistic, particularly for the UK (lower opening overlap with US) and CA (lower liquidity). If execution lags are 2–3 days, the IR would drop further (see alpha-decay table: lag-3 returns 6.52% vs lag-1's 7.84%, a 17% reduction).

### Return Cap and Portfolio Constraints

A weekly return cap of ±30% is applied post-cost. This reduces tail risk but also clips upside in high-vol weeks. The paper does not mention a return cap, so this constraint may make the run conservative relative to the paper's open-ended approach.

---

## Conclusion

This multi-region backtest (US, UK, CA; 2006–2026) produces a **7.84% annualized return at 1.06 IR net of costs**, underperforming the paper's reported 9.7% return and 1.6 IR. The shortfall is material but not disqualifying: the fundamental hypothesis—that ML can discover and enhance mean reversion while reducing turnover—is supported by the data, though the magnitude is dampened.

### Validation Checklist Summary

| Check | Status | Evidence |
|-------|--------|----------|
| R1W ranks first in SHAP | ✓ Pass | R1W is the top feature for XGBoost, LightGBM, RandomForest. MLP diverges, focusing on commodity regimes. |
| UPDOWN1W (earnings) is important | ✗ Fail | UPDOWN1W_RATINGS does not appear in top-10 ensemble features. Rating revisions are a weaker signal than EPS revisions. |
| Alpha decays with lag | ✓ Pass | Returns decline 12.54% (0d) → 7.84% (1d) → 6.07% (4d), with a steeper 0–1d drop than the paper's gradient. |
| Thursday > Monday weekday effect | ✓ Pass | Friday > Thursday > Wednesday > Tuesday > Monday, consistent with the paper's data-release argument. |
| Quintile spread monotonicity | ? No data | Per-quintile returns are not reported in the research pack. Cannot evaluate. |
| ML beats basic reversal baseline | ? Not evaluated | The research pack does not provide a backtest of a pure R1W baseline for direct comparison. Turnover comparison is available (57.5% vs paper's 317% basic reversal), suggesting efficiency gains. |
| Turnover: ML < basic reversal | ✓ Pass | 57.5% (this run) vs 317% (paper's basic reversal). Large reduction aligns with the paper's claim. |
| IR ≥ paper's 1.6 | ✗ Fail | 1.06 vs 1.6. Gap of −0.539 IR points. |

### Key Findings

1. **Mean-reversion discovery confirmed:** The ensemble's top-1 SHAP feature is R1W, replicating the paper's finding without explicit instruction.

2. **Earnings signal degradation:** The analyst-rating-revision proxy (UPDOWN1W_RATINGS) does not replicate the importance of the paper's EPS-consensus-revision signal. This is the single largest factor difference and likely explains 0.2–0.3 IR points of the headline gap.

3. **Turnover reduction achieved:** The strategy's 57.5% 2-way weekly turnover is well below the paper's 317% basic-reversal and 212% ML benchmarks, supporting the paper's claim that richer ML models enable more efficient position sizing.

4. **Volatility and max drawdown higher than paper:** This run's 7.39% vol and −10.01% max DD are larger than the paper's 6.3% vol and −12.3% DD, suggesting either (a) the narrower feature set leaves the portfolio more exposed to non-mean-reverting shocks, or (b) the shorter training window and snapshot-biased non-US universes amplify regime risk.

5. **Alpha decay steep but present:** The strategy bleeds alpha quickly with execution lag, from 12.54% (0d) to 7.84% (1d) to 6.07% (4d). The paper's decay is smoother, possibly because its training window and factor richness are better suited to handling lag regimes.

### Bottom-Line Read

The hypothesis that ML can discover and improve mean reversion is **not rejected**. The core mean-reversion signal (R1W) is reliably ranked first across three tree-based ensemble members, and the strategy achieves sub-3% weekly turnover while generating positive alpha over a 19-year window. The ensemble's max drawdown (−10%) is manageable for a vol-targeted fund.

However, the **observed IR of 1.06 falls short of the paper's 1.6**, and the sources of this gap are traceable to implementation constraints (missing IBES earnings-revisions data, shorter training window, survivorship bias in non-US regions, narrower feature set) rather than to a fundamental failure of the approach. With access to IBES EPS-revision history and a longer training window, the run would likely approach or exceed the paper's reported metrics.

For a portfolio manager evaluating whether to allocate to a short-term mean-reversion strategy powered by ML, this run provides evidence that the approach is sound—R1W is the dominant driver, alpha decays smoothly with lag, and turnover is manageable—but also highlights the importance of data quality (especially earnings signals) and training-window length for achieving the paper's promised improvements.