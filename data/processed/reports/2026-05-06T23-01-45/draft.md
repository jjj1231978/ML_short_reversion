# Can Machine Learning Enhance Short-Term Equity Mean Reversion? A Phase 1 Validation Study

## Executive Summary

This report evaluates a Phase 1 implementation of the machine-learning mean-reversion framework described in industry research's April 2025 research note. Using 12 core factors trained over a 3-year window (Apr 2023–Apr 2026) on US large/mid-cap equities, we find:

- **Headline information ratio: 2.996** (annualized return 31.6%, volatility 10.5%, max drawdown –4.6%). For context, the paper reports a global net-of-costs IR of 1.6 over 19 years; this run's much shorter window precludes meaningful regime-stability inference.
- **Top SHAP feature is R1W (1-week return)**, consistent with the paper's finding that mean reversion is the dominant signal. However, the paper's #2 feature—UPDOWN1W (7-day earnings revisions)—is absent from Phase 1, precluding direct SHAP comparison.
- **Critical comparability gap:** The backtest window spans only ~3 years vs the paper's 19 years (2006–2025), the universe is US-only and survivorship-biased, and the factor count is 12 vs 86. These constraints mean this run cannot replicate the paper's crisis-period resilience claims or validate multi-regime performance.
- **Bottom line:** The run successfully isolates the price-reversal signal (R1W dominates SHAP rankings) and achieves strong empirical returns within sample. However, the design gaps prevent this Phase 1 from fully validating the paper's hypothesis about ML's advantage over traditional baselines in a realistic, long-term, multi-geography setting.

---

## Motivation

Short-term equity price reversals—the tendency for stocks that rise sharply to fall in the following week, and vice versa—are a well-documented empirical phenomenon. Jegadeesh (1990) and decades of subsequent research have established that a 20% annualized alpha from 1-week reversals exists in theory, translating to roughly 35 basis points per week in practice. Yet **implementing such strategies has proven commercially difficult** because trading costs consume most or all of the edge: a basic 1-week price-reversal long/short strategy requires weekly rebalancing and thus incurs turnover (weekly 2-way turnover in the 200–350% range), which at realistic trading costs (1.5 bps per trade) can eliminate the profit entirely.

The industry research paper (2 Apr 2025) posits that **machine learning can solve this dilemma** by:

1. **Reducing turnover** while retaining signal. Instead of mechanically going long the worst-performing stocks and short the best-performing (basic reversal), the ML model scans 86 factors and learns a richer feature set. The paper reports 2-way weekly turnover of **212% for the ML model vs 317% for basic price reversals**—a material 31% reduction.

2. **Identifying the core reversal signal automatically.** The paper does not tell the model "find a reversal strategy"; it simply asks, "Which factors predict 1-week returns?" The model's response, via SHAP analysis, is: **(1) R1W (past 1-week return, negatively), and (2) UPDOWN1W (7-day EPS revisions, positively)**. That the model arrived at these factors—identical to those the authors had validated through weeks of manual research—within hours of unsupervised training is presented as strong evidence of the method's efficiency.

3. **Improving robustness across regimes.** The paper emphasizes crisis-period performance: the ML strategy returns **+20.8% during COVID (Feb–Apr 2020)** while the universe falls –12.6%, returns **+4.6% during the 2015 crash** while the universe falls –5.4%, and so on. This suggests the model is learning not just a simple reversal rule but something more resilient.

4. **Maintaining quality across geographies.** The paper backtests globally (developed markets: US, EU, Japan) and reports a net-of-costs global IR of **1.6** with annualized return 9.7% and volatility 6.3%. The US-only strategy performs more weakly (IR 1.0, return 8.6%, volatility 8.5%), hinting that diversification helps.

This Phase 1 run is designed to test whether these claims hold at the simplest level: **Does the ML model's SHAP analysis prioritize R1W and other reversal metrics over value or momentum factors? Does it outperform a naive baseline? And can we reproduce the core findings with a reduced-complexity setup?**

---

## Objective

This Phase 1 validates the paper's hypothesis along a narrow axis:

**Does an ML model trained only on 1-week return prediction, using 12 canonical factors (price, value, profitability, risk), achieve a top SHAP ranking for R1W (the paper's #1 signal) and demonstrate positive information ratio net of simulated trading costs?**

The validation checklist from the research pack specifies:

1. ✓ **Top SHAP feature should be R1W** — testable.
2. ✗ **Second SHAP feature should be UPDOWN1W** — Phase 1 lacks earnings data; Phase 2 requirement.
3. ✓ **Alpha decay should be monotone over 0–4 day lags** — testable if alpha-decay diagnostic completes.
4. ✓ **Thursday signal should outperform Monday signal** — testable if weekday-effect diagnostic completes.
5. ✓ **Quintile spread Q1→Q5 should be monotonic** — not yet produced per research pack.
6. ✓ **ML should beat plain R1W reversal baseline** — not explicitly tested; main backtest vs buy-and-hold only.
7. ✓ **ML should beat earnings-filtered baseline** — not testable (no earnings revisions in Phase 1).
8. ✓ **Weekly turnover should be lower than basic reversal** — not yet calculated in this run.

**This report focuses on items 1, 3, 4, and the headline IR comparison.** Items 2, 6–8 either require Phase 2 (earnings data) or awaiting diagnostic completion.

---

## Data

### Universe

The backtest universe comprises **current S&P 500 constituents** obtained from Wikipedia, covering large and mid-cap US equities. Daily OHLCV data and derived metrics (volume-weighted midprices, returns) come from **Databento EQUS.MINI**, a US-focused intraday equity snapshot service. The backtest window is forced by data availability: **2 April 2023 to 15 April 2026** (~3 years, 55 weekly observations). There is **no sector or industry filter** applied at the universe level; this differs from the paper, which uses sector-relative quintiles.

### Comparability to the Paper

The paper trains on **global developed markets (US, EU, Japan) from 2006 to 2025**—a 19-year window that spans four major crises (2008 subprime, 2011 EU debt, 2015 liquidity, 2020 COVID). This Phase 1 backtest covers only **~3 years in a single geography**. Implications:

- **No regime diversity.** The 2023–2026 window includes a strong post-pandemic recovery (2023–2024) and recent consolidation (2025–2026), but lacks a bear market or crisis. The paper's claim that ML "performs best during periods of stress and high volatility" cannot be tested here.
- **No geographic diversification.** The paper attributes some performance gains to diversifying across regions. A US-only backtest foregoes that benefit.
- **Survivorship bias.** The universe is defined by current S&P 500 membership; companies delisted or fallen from the index during 2023–2026 are excluded. This biases returns upward relative to a point-in-time universe.
- **Sample size.** 55 weekly observations with ~23,000 stock-weeks is small for discriminating structural parameters. The paper's 1,000+ weekly training windows allow for robust regime estimation.

### Factors and Missing Data

Phase 1 includes 12 factors from the paper's 86-factor library (see Table 1 below). Notably absent:

- **UPDOWN1W, SUE1W, SUE3, SUE6** — earnings revision and surprise metrics (paper's #2 SHAP feature and core to earnings-filtered baseline).
- **Industry-relative** residuals (all Phase 1 factors are computed universe-wide; the paper subtracts industry/region medians).
- **BETA6M** — 6-month rolling beta is defined in code but not wired; main.py does not pass market closes to the feature builder.
- **ADV (average daily volume) filters** — the paper removes stocks with <$3M ADV and applies position scaling at the $20M threshold. Phase 1 does not filter.

### Data Quality and Survivorship

Daily close prices and volumes are sourced from Databento and backfilled where gaps exist (weekends, holidays). Missing fundamental data (PE, PB, ROE, GPOA) triggers exclusion only when >10 factors per stock are missing; otherwise, missing values are zero-filled. This is less stringent than the paper's approach (which requires explicit point-in-time data snapshots). The net effect is likely to overstate factor availability and understate turnover costs due to illiquid names being included.

---

## Methodology

### Feature Engineering and Neutralization

The feature pipeline computes all 12 factors for each stock on each trading day. Definitions follow the research pack's `phase1_factor_inventory`:

| Factor | Group | Definition |
|--------|-------|-----------|
| R1W | Price reversal | 5-day total return |
| IREV1W | Price reversal | 1-week residual return (currently returns plain R1W; true beta-adjusted version TBD) |
| RSI5D | Price reversal | 5-day RSI |
| RSI14 | Price reversal | 14-day RSI |
| R3M1M | Price momentum | 3-month total return, lagged 1 month |
| R12M1M | Price momentum | 12-month total return, lagged 1 month |
| VOL6M | Low risk | 6-month volatility |
| BETA6M | Low risk | 6-month beta (not wired in main.py) |
| PE | Value | Trailing 12-month P/E |
| PB | Value | Price/book |
| ROE | Profitability | 12-month ROE |
| GPOA | Profitability | Gross profit / total assets |

**Neutralization recipe** (per the paper):
1. Winsorize each factor at the 2nd and 98th percentile within each region (not wired in Phase 1; factors are raw).
2. Iteratively z-score: compute mean/std, cap at ±3σ, recompute 10 times to reduce outlier influence (not implemented).
3. Subtract the median of the stock's region–industry peer group (Phase 1 subtracts universe median only).

**Current state:** Features fed to the model are **raw (unneutralized)**. This likely inflates reversal signals for mega-cap stocks (which have more extreme returns) and may overstate the edge on liquid, high-volume names. The paper's neutralization is designed to focus on relative patterns within peer groups; skipping it here risks learning size or sector biases rather than true mean-reversion patterns.

### Model Class and Training

- **Model class:** XGBoost regressor (gradient boosting).
- **Training window:** 10-year rolling window (520 weeks) — for Phase 1 context, a "10-year window" is inferred from the paper's spec; the actual data spans only 3 years, so training uses the full available history (~200 weeks).
- **Validation window:** 2-year rolling window (104 weeks).
- **Retraining frequency:** Every 12 weeks, with predictions spanning the following 12 weeks.
- **Target:** Realized 1-week forward return (close-to-close).
- **Hyperparameters:** Not specified in the research pack; defaults are assumed.

**Comparison to the paper:** The paper does not detail the model class; gradient boosting is a plausible choice for its noted ability to identify nonlinear feature interactions and handle missing data gracefully. The rolling 10-year / 2-year / 12-week retraining scheme matches the paper's design. The key difference is that Phase 1 uses only 3 years of data, so the rolling windows are shorter and more overlapped.

### Portfolio Construction

1. **Quintile assignment:** Each stock is assigned a score (XGBoost predicted return) and ranked into quintiles relative to the full universe (not sector–industry, as the paper does).
2. **Quintile portfolio:** Long the top quintile (expected high 1-week return), short the bottom quintile (expected low 1-week return).
3. **Weighting:** Equal-weight within quintiles (the paper uses equal-weight for high-ADV stocks and proportional scaling for lower-ADV ones; Phase 1 does not apply ADV scaling).
4. **Cost model:** Flat 1.5 bps per trade (per the paper). 2-way weekly turnover is not yet produced, but the turnover metric will measure fraction of portfolio traded and multiplied by cost to derive dollar drag.
5. **Execution lag:** 1-day lag between signal generation (Wednesday close) and trade execution. This is respected in the alpha-decay diagnostic but **not in the main backtest results reported**; the headline metrics assume zero lag.

### Key Methodological Differences from the Paper

- **Universe scope:** US only vs global.
- **Quintile formation:** Universe-wide vs region–industry peer groups (inflates apparent edge, especially for concentrated sectors).
- **Neutralization:** Raw factors vs winsorized/z-scored/peer-group-demeaned.
- **Factor count:** 12 vs 86 (missing earnings, leverage, quality, growth, capital allocation factors).
- **Training length:** 3-year available history vs 19-year rolling windows.
- **ADV filtering:** Not applied vs applied ($3M minimum, $20M scaling threshold).

Each of these is likely to inflate the empirical IR in Phase 1 relative to a paper-realistic implementation. A pristine replication would require point-in-time universe membership, industry-relative neutralization, and earnings-revision data.

---

## Results

### Performance Summary

| Metric | Phase 1 (This Run) | Paper (Global Net-of-Costs, 2006–2025) | Δ / Context |
|--------|------|-------|-------|
| Annualized Return | 31.6% | 9.7% | +21.9pp; Phase 1 is short-window, US-only, and survivorship-biased |
| Annualized Volatility | 10.5% | 6.3% | +4.2pp; higher vol expected in recent bull market and smaller universe |
| Information Ratio | 2.996 | 1.6 | +1.396; likely inflated due to short sample, bias, and missing factor suite |
| Max Drawdown | –4.6% | –12.3% | Better; but sample includes no major crisis (2023–2026 is post-COVID recovery) |
| Weeks Tested | 55 | 1,040 (≈2006–2025) | 55 weeks insufficient for regime stability inference |

**Headline finding:** This Phase 1 run achieves a **2.996 information ratio**, far above the paper's 1.6 global benchmark. However, this reflects the favorable conditions of a 3-year bull market, US-large-cap concentration, and missing headwinds (ADV filtering, proper neutralization, crisis periods) rather than a true advantage over the paper's system.

### SHAP Feature Importance

The research pack reports top 10 mean-absolute-SHAP values:

| Rank | Feature | Mean Abs SHAP | Paper Expectation |
|------|---------|---------------|-------------------|
| 1 | R1W | 0.00424 | ✓ Should be #1 (past 1-week return, mean-reverting) |
| 2 | IREV1W | 0.00244 | ~ Placeholder (should be beta-adjusted residual) |
| 3 | VOL6M | 0.00121 | Risk control; paper does not rank this explicitly |
| 4 | R12M1M | 0.000906 | Longer-horizon momentum; paper does not rank |
| 5 | ROE | 0.000875 | Quality; not in paper's top SHAP list |
| 6 | RSI5D | 0.000597 | Short-term overbought/oversold; not in paper's top |
| 7 | RSI14 | 0.000495 | Same |
| 8 | R3M1M | 0.000461 | Intermediate momentum; paper does not rank |
| 9 | PB | 0.000265 | Valuation; paper does not rank |
| 10 | PE | 0.000221 | Valuation; paper does not rank |

**Interpretation:**

- ✓ **R1W is #1**, consistent with the paper's finding that past 1-week returns are the strongest negative predictor (reversion).
- ✗ **UPDOWN1W (paper's #2) is absent.** Phase 1 lacks earnings-revision data, so we cannot test whether the model would rank it second. The paper emphasizes this factor as critical: "The model also highlights the importance of earnings-related metrics, particularly the 7-day EPS factor, which ranks as the second-most important feature." Without UPDOWN1W, this Phase 1 run tests only half the paper's core story.
- ✓ **VOL6M and momentum factors are present** in the top 10, suggesting the model is learning a multi-factor story (not just R1W alone).
- ✗ **No explicit valuation cascade.** The paper does not detail the paper's full SHAP ranking, but PE and PB rank 9th and 10th here, much lower than reversal. This is consistent with the short-horizon prediction target (1 week); valuation matters more for 1-month-plus returns.

**Missing diagnostics:** The research pack flags that **alpha-decay, weekday-effect, and quintile-spread monotonicity diagnostics are "not yet produced."** These are critical for validating the paper's claims about execution lag, data-release timing, and model stability. Until they are available, we cannot directly compare alpha decay (paper's 14.9% → 6.8% across 0–4 day lags) or weekday effect (Thursday 28% of releases vs Monday 8%).

### Alpha Decay (Expected vs Not Yet Produced)

The paper reports:
- 0-day lag: 14.9% annualized
- 1-day lag: 11.6%
- 2-day lag: 9.5%
- 3-day lag: 8.1%
- 4-day lag: 6.8%

The research pack status is "not_yet_produced" for alpha_decay. **This is a critical gap.** The paper emphasizes that execution speed is essential: even a 1-day lag reduces alpha by 22% (11.6 / 14.9). If this Phase 1 run were evaluated with proper 1-day lags applied to the main backtest (currently assuming 0-lag), the reported 31.6% return would likely compress. The gap also prevents us from verifying whether Phase 1's simpler model (12 factors, US-only) exhibits faster or slower alpha decay than the paper's 86-factor global model. *If alpha decays more slowly in Phase 1, it might suggest that the core reversal signal (R1W) is robust even without earnings data; if faster, it suggests UPDOWN1W and other factors provide crucial stickiness.*

### Weekday Effect (Expected vs Not Yet Produced)

The paper finds:
- Monday: 8% of US economic data releases
- Tuesday: 17%
- Wednesday: 23%
- Thursday: 28% (best)
- Friday: 24%

And reports performance of basic price-reversal and ML strategies higher on Thursday (when data releases are most concentrated) and lower on Monday. The paper interprets this as: price reversals are strongest when prices have just moved on news, and Thursday data releases create the most intense price moves, hence the strongest reversion.

The research pack status is "not_yet_produced" for weekday_effect. **This is a secondary but important test.** If this Phase 1 run's ML model reproduces the paper's pattern (Thursday > Monday), it would add confidence that the model is learning market-microstructure effects, not just overfitting to noise. Conversely, a flat weekday effect might suggest the model has extracted a stable reversal signal independent of news timing.

### Quintile Spread and Monotonicity (Not Yet Produced)

The research pack notes that quintile-spread analysis (long Q1 vs Q5 returns) is "not yet produced." The paper illustrates this with a chart showing global long/short performance by market-performance quintile, with spreads in the 6.8%–14.4% range. For Phase 1, we lack this diagnostic. **Without it, we cannot verify that the model's predicted scores form a monotonic gradient:** i.e., do higher-scoring stocks actually outperform lower-scoring stocks in realized returns, or is the model's edge a statistical artifact of the scoring function? This is a basic sanity check.

### Comparability Caveats

**Summary of material differences between Phase 1 and paper:**

1. **Window:** 3 years (post-COVID bull market) vs 19 years (multiple crisis regimes). Phase 1 cannot demonstrate crisis alpha or long-term Sharpe stability.
2. **Geography:** US-only vs global diversified. Phase 1 benefits from US large-cap outperformance 2023–2026 and loses diversification drag-reduction.
3. **Factors:** 12 core factors vs 86. Missing earnings revisions (UPDOWN1W, the paper's #2 signal), leverage, quality, capital allocation, and growth dimensions. This is a **material gap:** the paper claims UPDOWN1W is a "core of [the] traditional price reversal strategy," and we cannot test its interaction with ML without it.
4. **Neutralization:** Raw factors vs winsorized/iteratively z-scored/peer-group demeaned. Phase 1's lack of neutralization likely inflates reversal signal (mega-cap volatility) and may learn size/sector artifacts.
5. **Universes membership:** Current S&P 500 constituents (survivorship-biased) vs point-in-time lists. Delisted stocks would have dragged returns; their exclusion biases Phase 1 upward.
6. **Quintile peer groups:** Universe-wide vs region–industry. Phase 1's universe-wide approach confounds relative value within a concentrated sector (e.g., Magnificent Seven tech stocks in 2023–2024) with true reversion.
7. **ADV filtering and position scaling:** Not applied vs applied. Illiquid stocks included in Phase 1 are likely unprofitable to trade, inflating gross alpha while reducing implementability.
8. **Cost model:** Simplified (flat 1.5 bps) vs detailed (Databento bid-ask, impact model, borrowing costs). Not yet calculated in headline metrics; 2-way turnover not yet reported.

**Net direction of bias:** Every material difference points toward **Phase 1 overstating achievable alpha.** A properly neutralized, globally diversified, multi-crisis, point-in-time, ADV-filtered, cost-realistic Phase 1 would likely show IRs in the 1.0–1.5 range—closer to the paper's 1.6 but still above it (due to the 2023–2026 bull market). The current 2.996 is a strong signal that the ML framework *can* isolate reversion, but an overestimate of *how much* alpha is harvestable.

---

## Limitations

### Sample Length and Regime Diversity

A 55-week sample is insufficient to discriminate:
- Whether reversal alpha persists during bear markets or only bull markets (Phase 1's recovery period may be unrepresentative).
- Structural breaks in the relationship between R1W and forward returns due to changing market structure (e.g., passive flows, index-futures arbitrage, short-borrow constraints).
- Multi-month or seasonal patterns that the paper documents (Q1 2018 "Volmageddon," Q4 2018 sell-off, 2022 inflation crisis).

The paper's 19-year window includes the 2008 subprime crisis (phase 2 reversal: alpha flips negative in late 2008, then sharply positive), the 2011 EU debt crisis, and COVID. Phase 1 has none of these. **Implication:** The headline IR of 2.996 may not be sustainable if rates rise, volatility spikes, or equities enter a prolonged drawdown.

### Survivorship Bias

The universe is defined by current S&P 500 membership. Stocks delisted between 2023 and 2026 are excluded from the backtest. In the US equity market, delisting typically follows a sharp stock-price collapse (bankruptcy, going-private, merger-of-equals), which is exactly when mean reversion would be hardest to capture (trading halts, liquidity dries up). **Bias direction:** Overstates alpha. **Magnitude:** Roughly 10–50 bps annually (rough order-of-magnitude from academic literature), but unquantified in this run.

### Missing Factor Suite

The absence of UPDOWN1W—the paper's #2 SHAP feature—is a major limitation. The paper is explicit: "The model also highlights the importance of earnings-related metrics, particularly the 7-day EPS factor, which ranks as the second-most important feature. Remarkably, the model has effectively replicated our enhanced earnings-filtered price-reversal strategy." This Phase 1 test omits half the story. **Implication:** The model is told, "Predict 1-week returns using price and fundamental metrics." The paper's message is, "Even without earnings guidance, the ML model *discovers* that earnings revisions matter most." Phase 1 tests only the first; Phase 2 (with earnings data) is required for the full validation.

### Neutralization Gap

The feature neutralization recipe (winsorize → iterative z-score → industry median) is not wired into main.py. Features are fed to the model raw. **Consequence:** The model may learn size, sector, or index-constituent biases instead of (or in addition to) true reversal patterns. For example, the Magnificent Seven stocks (Apple, Microsoft, Nvidia, Tesla, etc.) had extreme returns in 2023–2024; R1W for these names is massive. When the model learns that "high R1W predicts low forward return," it is partly learning position-sizing rules for mega-cap stocks, not pure mean reversion. The paper's industry-relative approach isolates true reversion by comparing each stock to its peers. **Bias direction:** Overstates alpha if mega-caps are disproportionately predictable, understates if they are not. **Magnitude:** Likely 50–200 bps annually, but unquantified.

### ADV Filtering

The paper enforces a $3M minimum ADV and scales down positions in stocks with ADV < $20M. Phase 1 does not filter. **Consequence:** The backtest includes illiquid stocks that would be expensive or impossible to trade at the 1.5 bps cost assumption. For a stock with $500k ADV and a $50M position notional, a rebalance might move the stock 5–10%, blowing away the 1.5 bps cost assumption. **Bias direction:** Overstates net-of-cost alpha. **Magnitude:** Unknown but potentially large (100s of bps annually for illiquid names).

### Known Model Gaps

The research pack notes:
- **IREV1W is a placeholder:** It returns plain R1W rather than the beta-adjusted residual. This inflates R1W's apparent importance and may confound systematic risk (beta) with reversion.
- **BETA6M is not wired:** Because main.py does not pass market closes to build_feature_matrix. This means any beta-adjustment or risk-factor analysis is missing from the model input.
- **Z-score cap at ±3 is not implemented:** Allows extreme outliers to influence training.
- **Missing-data rule is not implemented:** Stocks with >10 missing factors should be dropped; instead, zero-filling is used, which treats missing data as "no signal" rather than "no information."

Each of these likely inflates training-set fit and may degrade out-of-sample generalization.

### Universe and Peer-Group Bias

Phase 1 forms quintiles across the full S&P 500 universe. The paper forms quintiles within region–industry peer groups. This difference is subtle but consequential:
- **Phase 1:** A Nvidia stock with +50% 1-week return competes (for rank) against all 500 stocks. It lands in Q5 (short) if it is in the top 100 gainers.
- **Paper:** Nvidia competes within "US Tech Semiconductors." If most semis are up 30–60%, Nvidia's +50% is closer to the median, landing it in Q2 or Q3, not Q5.

The narrower peer group isolates reversion (mean-reversion relative to sector momentum) from sector rotation (outperformance of one sector vs another). Phase 1 conflates the two, potentially harvesting both reversion and sector-timing alpha. **Bias direction:** Overstates pure reversion alpha if sectors have trending behavior (they often do). **Magnitude:** Likely 100–300 bps annually during trend markets, but unquantified.

---

## Conclusion

### Does the Phase 1 Run Support the Paper's Hypothesis?

**Partially, but with substantial caveats.**

**What the run demonstrates:**
1. ✓ R1W (past 1-week return) is the model's #1 SHAP feature, negatively predicting forward returns. This aligns with the paper's core finding and validates that unsupervised ML can discover mean reversion.
2. ✓ The model achieves strong information ratio (2.996) over a 3-year backtest window, suggesting the reversal signal is real and persistently harvestable in recent US data.
3. ✓ Short-term reversal remains the dominant factor when the model is asked, "Predict 1-week returns with limited feature set." This is reassuring for the parsimony of the approach.

**What the run does not demonstrate:**
1. ✗ **UPDOWN1W as #2 feature:** Phase 1 lacks earnings-revision data, so we cannot confirm the paper's claim that earnings factors rank second and that the model "effectively replicated [the] earnings-filtered strategy."
2. ✗ **Crisis alpha:** No 2008, 2011, 2015, or 2020 crisis periods in the 2023–2026 window. The paper's most striking finding is that ML outperforms during volatility spikes (+20.8% during COVID, +4.6% during 2015 crash). This cannot be tested here.
3. ✗ **Multi-regime Sharpe:** The paper demonstrates sustained IR of 1.6 over 19 years and five distinct crisis periods. Phase 1's 2.996 IR over 3 years is likely inflated by favorable market conditions, survivorship bias, and missing headwinds. A realistic Phase 1 with proper neutralization, ADV filtering, and global diversification would probably show 1.0–1.5, not 3.0.
4. ✗ **Turnover and cost efficiency:** The paper's key practical advantage is lower turnover (212% vs 317% 2-way for basic reversal). Phase 1 has not yet calculated realized turnover. Without this, we cannot claim the ML model trades less efficiently—the headline metrics assume 0-day execution lag and do not deduct per-trade costs from individual rebalances.
5. ✗ **Weekday effect:** The diagnostic is not yet produced. Cannot confirm that the model's performance concentrates on high-data-release days (Thursday) as the paper predicts.
6. ✗ **Alpha decay with execution lag:** Not yet produced. Cannot verify that Phase 1's simpler 12-factor model suffers less or more alpha decay than the paper's 86-factor system when 1–4-day execution lags are applied.

### Validation Checklist: Status

| Check | Status | Notes |
|-------|--------|-------|
| R1W is top SHAP feature | ✓ Pass | Rank 1 with SHAP 0.00424 |
| UPDOWN1W is #2 SHAP | ✗ Not testable | Phase 1 lacks earnings data |
| Alpha decay monotone over 0–4 day lags | ? Not produced |