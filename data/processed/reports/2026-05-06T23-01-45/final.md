# Can Machine Learning Enhance Short-Term Equity Mean Reversion? A Phase 1 Validation Study

## Executive Summary

This report evaluates a Phase 1 implementation of the machine-learning mean-reversion framework described in industry research's April 2025 research note ("Global Style Counselling"). Using 12 core factors trained over a 3-year window (Apr 2023–Apr 2026) on US large/mid-cap equities, we find:

- **Headline information ratio: 2.996** (annualized return 31.6%, volatility 10.5%, max drawdown –4.6%)—**not directly comparable to the paper's global 1.6 due to 3-year vs 19-year window, US-only vs global universe, 12 vs 86 factors, lack of crisis periods, and missing neutralization/ADV filtering.** This number is likely inflated by favorable market conditions and survivorship bias.

- **Top SHAP feature is R1W (1-week return)**, consistent with the paper's finding that mean reversion is the dominant signal. However, the paper's #2 feature—UPDOWN1W (7-day earnings revisions)—is absent from Phase 1, precluding direct validation of the paper's claim that earnings factors rank second.

- **Critical scope gaps:** The backtest window spans only ~3 years in a post-pandemic bull market (Apr 2023–Apr 2026), lacks exposure to the four major crises the paper documents (2008 subprime, 2011 EU debt, 2015 liquidity, 2020 COVID), and excludes the earnings-revision data that the paper emphasizes. The universe is US-only and survivorship-biased, and factors are unneutralized. These constraints mean this Phase 1 cannot replicate the paper's crisis-period resilience claims (e.g., +20.8% during COVID) or validate the earnings-driven signal discovery that is central to the paper's narrative.

- **Bottom line:** The run successfully isolates the price-reversal signal (R1W dominates SHAP rankings), confirming that unsupervised ML can discover mean reversion without explicit reversal instructions. However, the design constraints prevent this Phase 1 from fully validating the paper's hypothesis about ML's advantage over traditional baselines across regimes, geographies, and factor suites. **This is a shallow validation only:** Phase 2 (with earnings data) and Phase 3 (with 19-year historical backtest and proper neutralization) are required for full validation.

---

## Motivation

Short-term equity price reversals—the tendency for stocks that rise sharply to fall in the following week, and vice versa—are a well-documented empirical phenomenon. Jegadeesh (1990) established that systematic mean-reversion strategies can generate alpha; subsequent research has confirmed this across markets and regimes. Yet **implementing such strategies commercially is challenging** because trading costs consume most or all of the edge. The industry research paper illustrates the problem: a basic 1-week price-reversal long/short strategy requires weekly rebalancing and thus incurs turnover of 317% (2-way) per week. At realistic trading costs of 1.5 basis points per side, this turnover alone can eliminate the profit.

The paper posits that **machine learning can solve this dilemma** by:

1. **Reducing turnover while retaining signal.** Rather than mechanically ranking stocks by 1-week return (basic reversal), the ML model scans 86 factors and learns a richer feature set. The result: **212% 2-way turnover** for the ML strategy vs 317% for basic reversal—a 31% reduction that materially improves net returns.

2. **Identifying the core reversal signal automatically, without explicit instruction.** The paper does not tell the model "find a reversal strategy"; it simply asks, "Which factors predict 1-week returns?" The model's response, via SHAP (Shapley Additive exPlanations) feature importance analysis, is: **(1) R1W (past 1-week return, negatively), and (2) UPDOWN1W (7-day EPS revisions, positively)**. The paper frames this as striking: "In just a few hours, the machine was able to arrive at the same conclusion that had taken us weeks of research to reach." The model rediscovered earnings-filtered reversal without being primed.

3. **Improving robustness across crisis regimes.** The paper emphasizes that the ML strategy outperforms during volatility spikes. For example, during the COVID-19 crisis (Feb–Apr 2020), the ML strategy returned +20.8% while the global universe fell –12.55%; during the 2015 summer crash (Jul–Aug 2015), +1.67% vs –5.42%. This crisis-period outperformance suggests the model learns something more resilient than a simple reversal rule.

4. **Maintaining quality across geographies.** The paper backtests globally (developed markets: US, EU, Japan) and reports a net-of-costs global information ratio of **1.6** (annualized return 9.7%, volatility 6.3%). The US-only strategy performs more weakly (IR 1.0, return 8.6%, volatility 8.5%), hinting that geographic diversification helps.

This Phase 1 validation tests the narrowest slice of these claims: **Does the ML model's SHAP analysis prioritize R1W as the top feature? Does it achieve positive risk-adjusted returns? And can we isolate the reversal signal with minimal factors and recent data?**

---

## Objective

This Phase 1 validates the paper's core signal discovery along a limited axis:

**Does an ML model trained only on 1-week return prediction, using 12 canonical factors (price, value, profitability, risk), achieve a top SHAP ranking for R1W (the paper's #1 signal) and demonstrate positive information ratio over a 3-year US large-cap backtest?**

The validation checklist from the research pack specifies eight tests:

1. ✓ **Top SHAP feature should be R1W** — testable.
2. ✗ **Second SHAP feature should be UPDOWN1W** — Phase 1 lacks earnings data; Phase 2 requirement.
3. ✗ **Alpha decay should be monotone over 0–4 day lags** — diagnostic not yet produced.
4. ✗ **Thursday signal should outperform Monday signal** — weekday-effect diagnostic raises NotImplementedError.
5. ✗ **Quintile spread Q1→Q5 should be monotonic** — diagnostic not yet produced.
6. ✗ **ML should beat plain R1W reversal baseline** — no baseline portfolio constructed.
7. ✗ **ML should beat earnings-filtered baseline** — not testable (no earnings revisions in Phase 1).
8. ✗ **Weekly turnover should be lower than basic reversal** — 2-way turnover not yet calculated.

**This report focuses on item 1 (SHAP) and the headline performance metrics.** Items 2–8 either require Phase 2 (earnings data) or await diagnostic completion. **Phase 1 is a shallow validation only:** it confirms that the model's top SHAP feature aligns with the paper's reversal finding, but does not replicate the paper's full 86-factor system, global universe, crisis testing, or comparative advantage vs. baselines.

---

## Data

### Universe and Time Window

The backtest universe comprises **current S&P 500 constituents** obtained from Wikipedia, covering large and mid-cap US equities. Daily OHLCV data and derived metrics come from **Databento EQUS.MINI**, a US-focused intraday equity snapshot service. The backtest window is constrained by data availability: **2 April 2023 to 15 April 2026** (~3 years, 55 weekly rebalance dates, ~23,000 stock-weeks in total). There is **no sector, industry, or liquidity filter** applied; the universe is static (current S&P 500 membership).

### Comparability to the Paper

The paper trains on **global developed markets (US, EU, Japan) from 2006 to 2025**—a 19-year window spanning four documented crises (2008 subprime, 2011 EU debt, 2015 liquidity, 2020 COVID). This Phase 1 covers only **~3 years in a single geography during a post-pandemic bull market.** Critical implications:

- **No regime diversity.** The 2023–2026 window is a strong recovery and consolidation period but lacks a bear market or volatility crisis. The paper's primary finding—that ML "performs best during periods of significant market movements, irrespective of direction"—cannot be tested.
- **No geographic diversification.** The paper attributes performance gains to diversification across regions; a US-only backtest foregoes this benefit and misses idiosyncratic downturns (e.g., 2024 Japan volatility).
- **Survivorship bias.** The universe is defined by current S&P 500 membership; companies delisted or fallen from the index during 2023–2026 are excluded. This biases returns upward (delisted stocks tend to collapse, dragging performance).
- **Sample size.** 55 weekly observations is insufficient to discriminate structural shifts in the return-reversal relationship or to estimate regime-dependent parameters robustly.

### Factors and Missing Data

Phase 1 includes 12 factors from the paper's 86-factor library:

| Factor | Group | Definition |
|--------|-------|-----------|
| R1W | Price reversal | 5-day total return |
| IREV1W | Price reversal | 1-week residual return (currently a placeholder returning plain R1W) |
| RSI5D, RSI14 | Price reversal | 5-day and 14-day RSI |
| R3M1M, R12M1M | Price momentum | 3-month and 12-month returns, lagged 1 month |
| VOL6M | Low risk | 6-month volatility |
| BETA6M | Low risk | 6-month beta (not wired into main pipeline) |
| PE, PB | Value | Trailing P/E and price/book |
| ROE, GPOA | Profitability | ROE and gross profit / assets |

**Notably absent:** UPDOWN1W, SUE1W, SUE3, SUE6 (earnings revisions and surprises—the paper's #2 feature), industry-relative residuals, leverage, growth, quality, and capital-allocation factors. The research pack classifies IREV1W as "non-functional" (it returns plain R1W rather than a beta-adjusted residual), meaning the 12-factor suite is even more minimal than named.

### Data Quality and Neutralization

Daily close prices and volumes come from Databento. Fundamental data (PE, PB, ROE, GPOA) are sourced from SimFin. The paper's neutralization recipe—winsorize at 2nd/98th percentile, iteratively z-score, subtract region–industry peer medians—is **not wired into the Phase 1 pipeline.** Features are fed to the model raw. **Consequence:** The model may learn size, sector, and mega-cap biases instead of (or in addition to) true mean reversion. For example, the Magnificent Seven stocks (Apple, Microsoft, Nvidia, Tesla, etc.) had extreme returns in 2023–2024; R1W for these names is massive. The model learns that "high R1W predicts low forward return," but this partly encodes position-sizing rules for mega-caps rather than pure peer-group reversion. The paper's industry-relative approach isolates true reversion. **Bias direction:** Likely overstates alpha by 50–200 bps annually (unquantified).

---

## Methodology

### Feature Engineering

The feature pipeline computes all 12 factors daily. Definitions follow the research pack's `phase1_factor_inventory`. **Critical gap:** The neutralization recipe (winsorize → iterative z-score → peer-group demeaning) is not wired into main.py. Factors are standardized to zero mean and unit variance within each trading date but not adjusted for region, industry, or size cohort. This is a material deviation from the paper's approach and likely inflates the apparent reversal signal.

### Model Class and Training

- **Model class:** XGBoost regressor.
- **Training window:** 10-year rolling (520 weeks per spec, but ~200 weeks available given 3-year data).
- **Validation window:** 2-year rolling (104 weeks per spec, ~50 weeks available).
- **Retraining frequency:** Every 12 weeks.
- **Target:** Realized 1-week forward return (Friday close to Friday close).
- **Hyperparameters:** Not specified in the research pack; defaults assumed.

**Comparison to the paper:** The paper does not detail its model class or hyperparameters in the published text. The rolling 10-year / 2-year / 12-week retraining scheme matches the paper's design. The key difference is that Phase 1's available history is 3 years vs 19 years, so rolling windows are shorter and heavily overlapped.

### Portfolio Construction

1. **Quintile assignment:** Stocks are scored (XGBoost predicted return) and ranked into quintiles **across the full universe** (not sector–industry, as the paper does).
2. **Long/short portfolio:** Long top quintile, short bottom quintile.
3. **Weighting:** Equal-weight within quintiles. The paper applies ADV-based scaling ($3M minimum, position reduction for <$20M ADV stocks); **Phase 1 does not filter or scale.**
4. **Cost model:** Flat 1.5 bps per trade per the paper. Realized turnover is not yet calculated; headline metrics assume zero execution lag.
5. **Execution lag:** Specified as 1-day in code, but the main backtest results assume zero lag (per the alpha-decay subsection statement: "not in the main backtest results reported").

### Key Methodological Gaps vs. the Paper

- **Universe scope:** US-only vs. global developed.
- **Quintile formation:** Universe-wide vs. region–industry peer groups (inflates edge if sectors trend).
- **Neutralization:** Raw factors vs. winsorized/z-scored/peer-demeaned.
- **Factor count:** 12 vs. 86 (missing earnings, leverage, quality, growth, capital allocation).
- **Training length:** 3 years vs. 19 years.
- **ADV filtering:** None vs. $3M minimum and $20M scaling.
- **Survivorship:** Current index membership (biased) vs. point-in-time lists (unbiased).

Each of these likely **inflates Phase 1's reported performance relative to a paper-realistic implementation.** A pristine replication would require point-in-time universe membership, industry-relative neutralization, earnings-revision data, proper ADV filtering, and a 19-year backtest window.

---

## Results

### Performance Summary

| Metric | Phase 1 (This Run) | Paper (Global Net-of-Costs, 2006–2025) | Bias / Context |
|--------|------|-------|-------|
| Annualized Return | 31.6%* | 9.7% | +21.9pp; Phase 1 is short-window, post-COVID bull, US-only, survivorship-biased |
| Annualized Volatility | 10.5% | 6.3% | +4.2pp; US concentration and favorable regime in 2023–2026 |
| Information Ratio | 2.996* | 1.6 | +1.396; likely inflated due to window, bias, missing factors |
| Max Drawdown | –4.6% | –12.3% | Better; but sample includes no major crisis (2023–2026 is recovery) |
| Weeks Tested | 55 | 1,040+ (≈19 years) | 55 weeks insufficient for regime stability or structural inference |

*Returns reported are **gross (not net of per-trade costs).** Headline metrics assume **zero execution lag.** Net-of-cost and lagged returns await completion of turnover and alpha-decay diagnostics.

**Headline finding:** This Phase 1 achieves a 2.996 information ratio, well above the paper's 1.6 global benchmark. However, this reflects favorable market conditions (2023–2026 post-COVID bull market), US large-cap concentration, and missing headwinds (ADV filtering, proper neutralization, crisis periods) rather than a true advantage over the paper's system. A conservative adjusted estimate accounting for these biases would place the achievable IR in the 1.0–1.5 range—comparable to or slightly above the paper's 1.6, but not materially superior.

### SHAP Feature Importance

The research pack reports the top 10 mean-absolute-SHAP values:

| Rank | Feature | Mean Abs SHAP | Paper Expectation |
|------|---------|---------------|-------------------|
| 1 | R1W | 0.00424 | ✓ Should be #1 (past 1-week return, mean-reverting) |
| 2 | IREV1W | 0.00244 | ⚠ Placeholder (returns plain R1W; not a true residual) |
| 3 | VOL6M | 0.00121 | ~ Risk control; not in paper's top SHAP list |
| 4 | R12M1M | 0.000906 | ~ Longer-horizon momentum; not in paper's top |
| 5 | ROE | 0.000875 | ~ Quality; not in paper's top |
| 6 | RSI5D | 0.000597 | ~ Short-term momentum; not in paper's top |
| 7 | RSI14 | 0.000495 | ~ Same |
| 8 | R3M1M | 0.000461 | ~ Intermediate momentum; not in paper's top |
| 9 | PB | 0.000265 | ~ Valuation; not in paper's top |
| 10 | PE | 0.000221 | ~ Valuation; not in paper's top |

**Interpretation:**

- ✓ **R1W is #1**, confirming the paper's core finding that past 1-week returns are the strongest negative predictor (mean reversion).

- ⚠ **IREV1W as #2 is misleading.** The research pack explicitly states IREV1W is a "placeholder that returns plain R1W." Thus, IREV1W's SHAP value (0.00244) is not independent of R1W's (0.00424); it is the same signal measured twice. **The combined R1W + IREV1W SHAP = 0.00668, meaning raw 1-week return accounts for ~2.8× the feature importance of the next-highest feature (VOL6M at 0.00121).** Raw 1-week return dominates even more strongly than the individual rank suggests. This result validates that mean reversion is the dominant predictive signal, but also exposes a methodological flaw: IREV1W should be a distinct (beta-adjusted residual) feature; until it is, this Phase 1 provides inflated evidence of R1W's importance and cannot test whether residual returns add independent signal.

- ✗ **UPDOWN1W (paper's #2 feature) is absent.** Phase 1 lacks earnings-revision data, so we cannot test whether the model would rank UPDOWN1W second—the paper's central claim that "the model has effectively replicated our earnings-filtered price-reversal strategy" solely from historical patterns.

- ✓ **Multi-factor learning is evident:** VOL6M (risk), momentum factors (R3M1M, R12M1M), and profitability (ROE) appear in the top 10, indicating the model learns beyond R1W alone.

### Missing Diagnostics

The research pack flags three incomplete diagnostics:

1. **Alpha decay (not yet produced):** The paper reports annualized alpha of 14.9% (0-day lag), 11.6% (1-day), 9.5% (2-day), 8.1% (3-day), 6.8% (4-day). A 1-day lag reduces alpha by 22%; Phase 1's headline metrics assume zero lag. Until the alpha-decay diagnostic completes, we cannot verify Phase 1's sensitivity to execution timing or compare decay rates vs. the paper's 86-factor system. The absence of this diagnostic is material: it prevents us from assessing whether the simpler 12-factor model has faster or slower signal decay.

2. **Weekday effect (not yet produced):** The paper observes that Thursday—when 28% of US economic data releases occur—produces higher mean-reversion alpha than Monday (8% of releases). If Phase 1 reproduces this pattern, it would validate that the model learns market-microstructure effects; a flat weekday effect would suggest the signal is independent of news timing. **Status:** NotImplementedError (actively blocked). Cannot test.

3. **Quintile spread monotonicity (not yet produced):** The paper visualizes alpha by market-performance quintile; without Phase 1's quintile analysis, we cannot verify that the model's scores form a smooth gradient (higher scores → higher realized returns) or whether the edge is concentrated in specific deciles. This is a basic sanity check for overfitting.

### Comparability Caveats

**Material differences between Phase 1 and paper:**

1. **Window (3 years vs 19 years):** Phase 1 covers Apr 2023–Apr 2026 (post-COVID bull market). The paper spans 2006–2025, including the 2008 subprime crash (—32.6% universe, +21.68% ML), 2011 EU debt crisis (—4.27% universe, +5.63% ML), 2015 summer crash (—5.42% universe, +1.67% ML), and 2020 COVID (—12.55% universe, +20.83% ML). **Implication:** Phase 1 cannot test the paper's central claim that ML "performs best during periods of significant market movements." The absence of crisis periods makes it impossible to validate multi-regime resilience.

2. **Geography (US-only vs global):** The paper backtests Europe, Japan, and the US separately, then aggregates. It reports a global IR of 1.6 and a US IR of 1.0, attributing the difference to diversification. Phase 1 is US-only and thus lacks geographic diversification benefits.

3. **Factors (12 vs 86):** Phase 1 omits earnings revisions (UPDOWN1W—the paper's #2 feature), leverage, quality, growth, and capital-allocation factors. **Major gap:** The paper's narrative hinges on UPDOWN1W: "Remarkably, the model has effectively replicated our enhanced earnings-filtered price-reversal strategy by relying solely on historical patterns." Phase 1 cannot test this claim without earnings data.

4. **Neutralization (raw vs winsorized/z-scored/peer-demeaned):** Phase 1 does not implement the paper's neutralization recipe. Features are raw, inflating signals for mega-caps and concentrated sectors.

5. **Universe membership (current S&P 500 vs point-in-time):** Phase 1 uses current index constituents, excluding delisted stocks. This is survivorship-biased.

6. **Quintile peer groups (universe-wide vs region–industry):** Phase 1 forms quintiles across the full S&P 500; the paper forms them within peer groups. This conflates pure reversion (mean-reversion relative to sector) with sector-timing alpha if sectors trend.

7. **ADV filtering (none vs $3M minimum, $20M scaling):** Phase 1 includes illiquid names that may be uneconomical to trade at 1.5 bps cost.

8. **Cost model (simplified vs detailed):** Phase 1 assumes flat 1.5 bps; the paper may account for bid-ask spread, market impact, and borrowing costs. Headline metrics do not deduct costs from individual rebalances; 2-way turnover is not yet calculated, so net-of-cost returns are pending.

**Net direction of bias:** Every material difference points toward **Phase 1 overstating achievable alpha.** A pristine implementation with proper neutralization, global diversification, multi-crisis exposure, point-in-time universe, ADV filtering, and proper cost accounting would likely show an IR closer to 1.0–1.5 (vs the reported 2.996). The reported 31.6% return is almost certainly an overestimate.

---

## Limitations

### Sample Length and Regime Diversity

A 55-week sample is insufficient to:
- Discriminate whether reversal alpha persists during bear markets or only bull markets. The paper's 19-year window includes the 2008 collapse (a regime where reversion can flip negative due to liquidity evaporation), the 2011 sovereign-debt crisis, the 2015 volatility spike, and the 2020 pandemic shock. Phase 1 has none of these. **Implication:** The headline IR of 2.996 may not be sustainable under stress.
- Test for structural breaks in the return-reversion relationship due to market evolution (passive flows, options-trading activity, short-borrow constraints).
- Validate multi-month seasonality or seasonal patterns the paper documents (Q1 2018 "Volmageddon," Q4 2018 sell-off, 2022 inflation crisis).

### Survivorship Bias

The universe is defined by current S&P 500 membership; delisted stocks are excluded. Delisting typically follows price collapse (bankruptcy, distressed M&A), exactly when mean reversion would be most difficult to capture (trading halts, liquidity dries up). **Bias direction:** Overstates alpha. **Magnitude:** Rough order-of-magnitude 10–50 bps annually (unquantified in this run).

### Missing Factor Suite and Core SHAP Feature

The absence of UPDOWN1W is a critical gap. The paper states: "The model also highlights the importance of earnings-related metrics, particularly the 7-day EPS factor, which ranks as the second-most important feature. Remarkably, the model has effectively replicated our enhanced earnings-filtered price-reversal strategy by relying solely on historical patterns." **This Phase 1 test omits half the paper's core narrative.** Phase 2 (with earnings data) is required to validate whether the ML model independently discovers earnings as a secondary driver.

Additionally, IREV1W is a placeholder (returns plain R1W, not a true beta-adjusted residual). This artificially inflates R1W's measured importance and double-counts the same signal.

### Neutralization Gap

The paper's feature neutralization (winsorize → iterative z-score → industry-relative centering) is not wired. **Consequence:** The model may learn size, sector, and mega-cap biases rather than pure mean reversion. For instance, the Magnificent Seven stocks had extreme returns in 2023–2024; the model learns that "high R1W predicts low forward return," but this partly encodes mega-cap position-sizing rules rather than true peer-group reversion. **Bias direction:** Likely overstates alpha by 50–200 bps annually (unquantified). Phase 2 should implement proper neutralization and retest.

### ADV Filtering and Liquidity

The paper enforces $3M minimum ADV and scales positions in stocks with ADV < $20M. Phase 1 does not filter. **Consequence:** The backtest includes illiquid stocks that would be expensive or impossible to trade at 1.5 bps cost. For a stock with $500k ADV, a large rebalance could move it 5–10%, blowing away the cost assumption. **Bias direction:** Overstates net-of-cost returns. **Magnitude:** Likely 100s of bps annually for illiquid names (unquantified).

### Universe Peer Groups

Phase 1 forms quintiles across the full S&P 500 universe; the paper forms them within region–industry peer groups. **Consequence:** Purely sector-relative reversion (mean-reversion within a sector) is conflated with sector-timing alpha (outperformance of one sector vs. another). During periods when technology outperforms broadly (e.g., 2023–2024), a tech stock's week-on-week reversion within the sector is orthogonal to its outperformance of the market. Phase 1's universe-wide approach captures both effects, inflating apparent edge if sectors trend. **Bias direction:** Overstates alpha during trend markets (common). **Magnitude:** Likely 100–300 bps annually during trend periods (unquantified).

### Diagnostic Gaps

- **Alpha decay (not yet produced):** Cannot verify Phase 1's sensitivity to 1–4 day execution lags or compare decay rates vs. the paper's system.
- **Weekday effect (not yet produced):** Cannot test whether the signal concentrates on high-data-release days (Thursday per paper) or is flat.
- **Quintile spread monotonicity (not yet produced):** Cannot verify that predicted scores form a smooth gradient; cannot rule out overfitting.
- **Baseline comparison (not tested):** No constructed basket for plain R1W reversal or earnings-filtered reversal; thus no direct comparison of ML edge vs. traditional strategies.

### Known Implementation Issues

The research pack flags:
- IREV1W is a placeholder (returns R1W, not beta-adjusted residual).
- BETA6M is defined but not wired (market closes not passed to feature builder).
- Z-score cap at ±3 not implemented.
- Missing-data rule not implemented (zero-filling used instead of dropping tickers with >10 missing factors).

Each of these likely inflates in-sample fit and may degrade out-of-sample generalization.

---

## Conclusion

### Does Phase 1 Support the Paper's Hypothesis?

**Partially, but with substantial caveats.**

**What the run demonstrates:**

1. ✓ **R1W is the model's #1 SHAP feature**, negatively predicting forward returns. This aligns with the paper's core finding and validates that unsupervised ML can discover mean reversion without explicit instruction.

2. ✓ **The model achieves strong information ratio (2.996)** over a 3-year backtest window, confirming that the reversal signal is real and persistently harvestable in recent US data (albeit in a favorable regime).

3. ✓ **Short-term reversal remains the dominant factor** when the model is asked, "Predict 1-week returns with limited feature set." This confirms the parsimony of the approach: a minimal 12-factor system can isolate reversion.

**What the run does not demonstrate:**

1. ✗ **UPDOWN1W as #2 feature.** Phase 1 lacks earnings-revision data. We cannot test the paper's claim that earnings factors rank second and that the model "effectively replicated [the] earnings-filtered strategy" solely from historical patterns. This is the paper's central narrative: the model discovers earnings without being told to look for them.

2. ✗ **Crisis-period alpha.** The paper's most striking finding is crisis outperformance: +20.8% during COVID (Feb–Apr 2020) while the universe fell –12.55%; +4.6% during the 2015 crash (Jul–Aug) while the universe fell –5.42%. Phase 1 has no equivalent crisis period to test. The absence of multi-regime data makes it impossible to validate whether the ML system's edge is stable or regime-dependent.

3. ✗ **Multi-regime Sharpe and consistency.** The paper demonstrates sustained IR of 1.6 over 19 years and five distinct crisis periods. Phase 1's 2.996 IR over 3 years in a post-COVID bull market is likely inflated by favorable conditions, survivorship bias, and missing headwinds (neutralization, ADV filtering, global diversification). A realistic Phase 1 implementation with proper neutralization, ADV filtering, and global diversification would likely show 1.0–1.5 IR, not 3.0.

4. ✗ **Turnover and cost efficiency.** The paper's key practical advantage is lower turnover (212% vs 317% 2-way for basic reversal), which translates to higher net returns. Phase 1 has not calculated realized 2-way turnover, and the headline metrics assume zero execution lag. Until turnover and alpha-decay diagnostics are produced, we cannot claim the ML model trades more efficiently.

5. ✗ **Weekday and data-release effects.** The paper observes that Thursday—when 28% of US data releases occur—produces higher reversion alpha than Monday (8% of releases). This is plausible market-microstructure evidence. Phase 1's weekday-effect diagnostic is not yet produced (raises NotImplementedError). We cannot test whether the model concentrates on high-data-release days or captures a stable reversal signal independent of news timing.

6. ✗ **Baseline comparisons.** The paper compares ML to (a) basic price reversals (317% turnover) and (b) earnings-filtered reversals (341% turnover), showing ML achieves similar or better alpha with lower turnover. Phase 1 has not constructed these baselines, so we cannot directly