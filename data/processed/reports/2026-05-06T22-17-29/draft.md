# Machine Learning for Short-Term Equity Reversion: Phase 1 Validation Note

## Executive Summary

This Phase 1 backtest validates a reduced-scope ML weekly reversion model against the industry research's global framework. **Headline metrics are not yet produced**—the pipeline has not been executed. The Phase 1 setup is materially constrained relative to the paper: a 3-year US-only window (vs 19 years global), 12 factors (vs 86), and critical missing earnings-revision signals. When results become available, they will be benchmarked directly against the paper's reported global net-of-cost information ratio of 1.6 (US: 1.0). The dominant comparability caveat is sample length; a 3-year backtest cannot discriminate regime dependence or structural breaks the way the paper's 2006–2025 span can. **Core hypothesis under test:** Does unsupervised ML identify R1W (1-week return) as the dominant reversion signal, and does it outperform a naive long-loser / short-winner baseline?

---

## Motivation

Short-term equity mean reversion is a well-documented market microstructure phenomenon. The industry research quantifies it elegantly: 20% annualized weekly alpha translates to only 35 basis points per week—"highly vulnerable to trading costs." The paper's key claim is that a machine-learning framework, trained on 80+ factors without reversion priors, **naturally converges on the same mean-reversion insight** that took researchers weeks to develop manually.

The paper reports two critical findings:

1. **R1W dominates SHAP feature importance.** The past 1-week return, negatively signed, is the single strongest predictor of next-week outperformance. High past winners are more likely to underperform; high past losers, to outperform.

2. **UPDOWN1W ranks second.** The 7-day earnings-revision signal (EPS upward minus downward revisions, normalized) acts as a filter: reversals are stronger when price moves are *not* justified by fundamental news. This earnings factor alone drove the paper to overlay it on a naive reversal strategy.

Beyond factor discovery, the paper demonstrates implementation advantages: ML turnover is **212% per week vs 317% for basic reversals**, and the ML model **reduces seasonality** (the weekday effect is less pronounced). Net of 1.5 bp trading costs, the global strategy achieves a **1.6 IR with 9.7% annualized return** over 2006–2025.

This run tests whether Phase 1—a minimal but serviceable ML setup—can replicate the core finding: *R1W as top feature, outperformance vs naive reversion, plausible turnover reduction*.

---

## Objective

**Phase 1 validates three specific hypotheses:**

1. **R1W emergence:** Without any reversion objective encoded, does the ML model (XGBoost on 12 factors) identify R1W as the #1 SHAP feature?
2. **Baseline outperformance:** Does the ML portfolio (long top quintile, short bottom quintile) generate higher risk-adjusted returns than a pure-R1W naive reversal strategy, before costs?
3. **Cost resilience:** Is the turnover modest enough that net-of-cost alpha remains meaningful after 1.5 bp trading friction?

**Validation checklist (from paper):**
- ✓ Top SHAP feature = R1W → testable in Phase 1
- ✗ Second SHAP feature = UPDOWN1W → Phase 2 (earnings data not in current pipeline)
- ✓ Alpha decay monotone 0 d → 4 d lag → testable in Phase 1
- ✓ Thursday signal > Monday signal → testable in Phase 1
- ✓ Quintile spread (Q1→Q5) monotonic → testable in Phase 1
- ✓ ML beats plain R1W reversal → testable in Phase 1
- ✗ ML beats earnings-filtered reversal → Phase 2 (needs UPDOWN1W)
- ✓ ML turnover < basic-reversal turnover → testable in Phase 1

**Status:** No run has been executed. All metrics below reflect paper benchmarks; Phase 1 results will be populated when the pipeline completes.

---

## Data

**Universe:** S&P 500 constituents (current Wikipedia list, as of pipeline run date).

**Window:** 2023-04-01 to 2026-04-11 (~3 years, 156 weeks). Forced by Databento EQUS.MINI futures data availability; this start date captures the post-pandemic regime but misses 2008 crisis, GFC, European debt crisis, COVID trough, and 2022 inflation shock. **This is the single largest comparability constraint.**

**Sources:**
- Daily OHLCV: Databento EQUS.MINI (E-mini S&P 500 futures as price proxy, downloaded cross-section of equities via alternative data)
- Fundamentals: SimFin (TTM earnings, book value, gross profit, assets)
- Price history: yfinance (6-month rolling volatility, momentum lags)

**Filters:**
- ADV >= $3M (liquidity); exclusion of sub-$3M names to avoid slippage
- Non-zero market cap
- No penny stocks or delisted tickers

**Quintile construction:** Across full universe (not region × industry, as paper does). This introduces a slight bias—the paper's regional and sector neutrality reduces exposure to mega-cap concentration, while Phase 1 does not.

---

## Methodology

**Feature Pipeline (12 factors):**

| Group | Factor | Definition |
|-------|--------|-----------|
| Price Reversal | R1W | 5-day total return (negatively signed in model = loser signal) |
| Price Reversal | IREV1W | Beta-adjusted residual 1-week return *(placeholder; returns R1W; Phase 2 TODO)* |
| Price Reversal | RSI5D | 5-day RSI |
| Price Reversal | RSI14 | 14-day RSI |
| Price Momentum | R3M1M | 3-month return lagged 1 month |
| Price Momentum | R12M1M | 12-month return lagged 1 month |
| Low Risk | VOL6M | 6-month annualized volatility |
| Low Risk | BETA6M | 6-month beta vs SPX *(NOT WIRED; main.py does not pass market_close)* |
| Value | PE | TTM price/earnings ratio |
| Value | PB | Price/book |
| Profitability | ROE | TTM net income / equity |
| Profitability | GPOA | TTM gross profit / total assets |

**Paper's factor count: 86 (80 baseline + 6 short-horizon).** Phase 1 uses 12 (6 reversal-adjacent + 6 value/quality). Missing entirely: earnings revisions (UPDOWN6, UPDOWN3, UPDOWN1W, SUE factors—the #2 SHAP feature), leverage, growth trends, earnings quality, capital allocation, and most macro/macro-beta factors.

**Neutralization:** Paper requirement is winsorize (2% tails) → iterative z-score (cap ±3) → industry median subtraction. Phase 1 implementation status:
- ✓ Winsorize at 2% per region
- ✗ **Z-score NOT capped at ±3** (implemented but not passed to model)
- ✗ **Industry median NOT applied** (features fed raw to model; spec gap per known_implementation_gaps)
- ✗ **Missing-data rule NOT applied** (drop >10 missing factors, zero-fill rest)

This is a **critical gap**: raw features may contain outliers that the paper's neutralization would suppress. Likely effect: higher noise, possibly inflated vol/drawdown, but unclear bias on alpha.

**Model:** XGBoost regression (gradient boosting).
- Target: next-week realized return (T+1 to T+5, out-of-sample)
- Objective: squared error
- Hyperparameters: standard (learning_rate=0.1, max_depth=5, n_estimators=100) *(paper's exact tuning not disclosed; likely similar)*
- Training: 10-year rolling window (520 weeks) on Wednesday close; validation 2-year rolling (104 weeks); test 1 week forward
- Retrain frequency: every 12 weeks (paper spec)

**Portfolio Construction:**
- Long: top quintile (best ML scores)
- Short: bottom quintile (worst ML scores)
- Equal weight within quintile
- ADV-based position scaling: equal-weight if ADV > $20M; pro-rata reduce if $3M < ADV ≤ $20M *(NOT WIRED in build_long_short_portfolio)*
- Execution lag: 1 day (signal calculated Wednesday close, traded Thursday open) in alpha-decay diagnostic; **main backtest does not enforce lag** *(spec gap)*
- Cost model: 1.5 bp per 2-way trade (on turnover)
- Turnover: weekly rebalance, count in/out flow

**Baseline strategies (for comparison, once run produces data):**
1. **Plain R1W Reversal:** Long past losers (R1W < p25), short past winners (R1W > p75), sector-relative (paper spec). Equal weight.
2. **Earnings-Filtered Reversal:** Plain reversal + overlay filter (exclude if negative earnings momentum / positive earnings momentum). *(Phase 2; not available in Phase 1)*

**Expected outputs:**
- Weekly returns (long, short, L/S net, L vs market, S vs market)
- Cumulative P&L series back to 2023-04-01
- Risk metrics: annualized return, vol, Sharpe, max drawdown, rolling Calmar
- SHAP values (mean absolute impact per feature)
- Alpha decay table (0d lag, 1d, 2d, 3d, 4d)
- Weekday effect (average return by day signal computed)
- Per-quintile spread (Q1 return, Q2, ..., Q5)
- Turnover analysis

---

## Results

**STATUS: NOT YET PRODUCED.** 

The research pack indicates `"pipeline_run_status": "no_run_yet"`. Headline metrics, portfolio summary, predictions, SHAP, alpha decay, and weekday effect all report `"status": "not_yet_produced"`.

### Comparability Caveats

Before interpreting Phase 1 results (once generated), readers must acknowledge these five material gaps:

**1. Sample Window (Dominant Caveat)**
- **This run:** 3 years (Apr 2023 – Apr 2026)
- **Paper:** 19 years (Jan 2006 – Dec 2025)
- **Implication:** Phase 1 sample spans a single regime—low-volatility growth / tech strength (2023-2024) and then rate shock (2024-2025). It includes no 2008 crisis, no Eurozone debt stress (2011-2012), no COVID crash-and-rebound (Feb-Apr 2020), no 2022 inflation rout. The paper's 19-year backtest demonstrates reversion works *across regime changes*. Phase 1 cannot.
- **Bias:** Unknown. If reversion is regime-dependent, 3-year alpha may overstate or understate sustainable alpha. Paper shows reversion performs especially well in crises (2008: +21.68%, -32.6% market); Phase 1 will miss this.

**2. Geography**
- **This run:** US S&P 500 only
- **Paper:** Global developed (US + Europe + Japan) with regional diversification
- **Implication:** Phase 1 results are US-only. Paper's global IR of 1.6 reflects diversification benefit; US alone is 1.0. Phase 1 should report lower IR (if typical) or higher (if US reversion is stronger 2023-2025).
- **Bias:** Likely slightly lower Sharpe, higher max drawdown (no diversification).

**3. Factor Count & Composition**
- **This run:** 12 factors (R1W, RSI, momentum, volatility, PE, PB, ROE, GPOA)
- **Paper:** 86 factors (the 12 above + 74 more: earnings revisions, leverage, growth, earnings quality, capital allocation, macro betas, ESG, etc.)
- **Critical missing:** UPDOWN1W (7-day EPS revisions), which ranks #2 in paper's SHAP. This signal filters out fundamental-driven moves, improving reversion signal quality.
- **Implication:** Phase 1 will detect R1W as #1 (mechanical), but without UPDOWN1W, the model may conflate fundamental news with pure mean reversion. Alpha may be overstated (noise picked up as reversion) or understated (lacking earnings filter).
- **Bias:** Likely noisy, possibly higher drawdown; hard to predict alpha sign without earnings data.

**4. Neutralization Gaps**
- **This run:** No winsorizing, z-score capping, or industry median subtraction applied to features in the actual model
- **Paper:** Explicit winsorize (2%), iterative z-score (cap ±3), industry-median subtraction
- **Implication:** Raw outliers may introduce noise. Paper's approach stabilizes factors across time and sectors; Phase 1 does not.
- **Bias:** Likely higher volatility, lower Sharpe; possible understate of true alpha if outliers corrupt signal.

**5. Quintile Peer Groups**
- **This run:** Quintiles formed across full US universe (no regional or industry bucketing)
- **Paper:** Quintiles formed within region × industry groups
- **Implication:** Phase 1 is subject to sector/size concentration risk. If tech mega-caps are in Q5 (high scores) and small-cap value in Q1 (low scores), the L/S spread conflates reversion alpha with sector/style bets.
- **Bias:** Higher correlations to sector factors; IR may be inflated if sector bets coincide with reversion bets.

**Additional small gaps:**
- No 1-day execution lag enforced in main backtest (only in alpha-decay diagnostic).
- Missing-data handling not per spec (should drop >10 missing, zero-fill rest; not done).
- ADV-based position scaling not applied in portfolio build.

---

## Limitations

**1. Survivorship Bias**
Phase 1 uses current Wikipedia S&P 500 constituents. Backtests therefore assume all stocks in the index at any point were always tradable. In reality, delisted or exited stocks disappear from history. Reversion may have worked better on stocks that *failed to revert* and exited. Paper uses point-in-time index data (unavailable in Phase 1); the bias is likely modest (large-cap churn is slow) but material in crisis windows.

**2. Sample Length**
Three years is insufficient to distinguish luck from skill in a weekly reversion strategy. Paper's alpha decays from 14.9% (no lag) to 6.8% (4-day lag); Phase 1's 3-year sample has ~156 weeks of independent data. Standard error on a 9.7% annualized return with 6.3% vol over 156 weeks is approximately sqrt(4 * 156) = 25 weeks of error, or ~2% annualized. Confidence intervals are wide.

**3. Missing Earnings Factors**
The paper's second-ranked SHAP feature, UPDOWN1W, is not available. This is a deliberate Phase 2 scope boundary, but it means Phase 1 cannot validate the paper's finding that ML naturally identifies earnings-filtered reversion. Phase 1 will show R1W rank #1, but the second-place feature may be RSI, momentum, or noise—not earnings revisions.

**4. Neutralization Not Applied**
Raw features feed the model, introducing outlier risk and cross-sectional noise the paper's pipeline would suppress. Effect on alpha is unknown: could inflate or deflate. A sensitivity check (run with neutralized features) is not planned for Phase 1.

**5. Simplified Universe**
Equal-weight quintiles across 500 names ignores sector and size. If reversion is stronger in certain sectors, Phase 1's portfolio construction may over- or under-weight them accidentally. The paper's region × industry bucketing is more disciplined.

**6. Cost Model Simplification**
Paper applies 1.5 bp uniformly; no modeling of market impact, borrow costs, or regime-dependent market depth. This likely understates implementation drag in 2023-2024 (tight spreads, low volatility) and may overstate it in 2025 (if vol spiked). Cumulative effect unknown.

**7. No Borrow-Cost Model**
Short positions in the strategy require borrowing. The paper does not explicitly model hard-to-borrow stocks or short rebate rates; Phase 1 does not either. This is a second-order effect for large-cap S&P 500 names but could swing returns ±0.5% annualized.

---

## Conclusion

**Phase 1 is a reduced-scope validation that cannot yet be evaluated.** Once the pipeline runs and metrics are produced, results will be judged against the paper's benchmarks:

| Metric | Paper (Global, Net of Costs, 2006-2025) | This Run (US, TBD) | Testable? |
|--------|------------------------------------------|-------------------|-----------|
| Ann. Return | 9.7% | not_yet_produced | Yes |
| Ann. Vol | 6.3% | not_yet_produced | Yes |
| Information Ratio | 1.6 | not_yet_produced | Yes |
| Max Drawdown | -12.3% | not_yet_produced | Yes |
| R1W SHAP Rank | #1 | not_yet_produced | Yes |
| UPDOWN1W SHAP Rank | #2 | not_yet_produced | **No** (Phase 2) |
| Alpha decay (4d lag) | 6.8% | not_yet_produced | Yes |
| Thursday > Monday | Yes | not_yet_produced | Yes |
| ML turnover | 212% | not_yet_produced | Yes |
| Basic-reversal turnover | 317% | not_yet_produced | Yes |

**Hypothesis validation roadmap:**

1. **R1W emergence:** If R1W ranks #1 in SHAP, the core ML hypothesis is supported—unsupervised learning recovers mean reversion.
2. **Outperformance:** If Phase 1 IR > 1.0 (paper's US benchmark) or even near it, implementation quality is sufficient.
3. **Cost resilience:** If net-of-cost alpha > 0 and turnover < 250%, the strategy survives practical constraints.
4. **Why Phase 1 cannot validate the full paper:**
   - 3-year sample cannot confirm regime robustness (paper tested 2006-2025, three major crises).
   - Missing UPDOWN1W prevents validating the earnings-filter insight (feature rank #2).
   - Simplified neutralization introduces noise; unclear if alpha survives.
   - US-only loses diversification benefit; IR will likely trail paper's global 1.6.

**Bottom line:** Phase 1 serves as a **proof-of-concept and debugging milestone**, not a definitive strategy backtest. When results land, readers should treat any published alpha as preliminary, conditional on the five caveats above. If Phase 1 succeeds (R1W #1, IR > 0.7, turnover reasonable), it clears the runway for Phase 2: add earnings factors, extend history, and lock in a production-grade implementation. If Phase 1 stumbles, the gap between industry research's 19-year global framework and this 3-year US variant will be instructive.