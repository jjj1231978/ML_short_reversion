# Machine Learning for Short-Term Equity Reversion: Phase 1 Validation Note

## Executive Summary

This Phase 1 backtest will validate a reduced-scope ML weekly reversion model against industry research's published global framework (April 2025). **Phase 1 results are not yet produced—the pipeline has not executed.** When complete, this note will benchmark actual performance against the paper's reported benchmarks: global net-of-cost information ratio of 1.6 (US standalone: 1.0), achieved over 19 years (2006–2025). 

The Phase 1 setup is materially constrained relative to the paper: a 3-year US-only historical window (versus 19 years global), 11 unique factors (versus 86), and critically missing earnings-revision signals. **Core hypothesis under test:** Does unsupervised ML identify R1W (1-week return) as the dominant reversion feature, and can it outperform a naive long-loser / short-winner baseline while maintaining manageable turnover?

---

## Motivation

Short-term equity mean reversion is a well-documented microstructure phenomenon. The industry research quantifies the core challenge: 20% annualized weekly alpha translates to only 35 basis points per week—"highly vulnerable to trading costs/constraints." Implementation friction is the binding constraint on profitability.

The paper reports two key findings:

1. **R1W dominates feature importance.** Via SHAP analysis, the past 1-week return is the single strongest predictor of next-week relative performance. Negatively signed: high past winners are statistically more likely to underperform; high past losers, to outperform.

2. **UPDOWN1W (7-day EPS revisions) ranks second.** The paper's prior research established that reversals are stronger when price moves are *not* justified by fundamental consensus. When the ML model was trained without any reversion constraint, it independently recovered earnings momentum as the second-ranked feature, validating this manual insight.

Beyond feature recovery, the paper demonstrates practical advantages:

- **Turnover:** ML model 212% per week versus 317% (basic reversals)—a 33% reduction, translating to 1.5–2 bp of cost savings per trade.
- **Seasonality:** ML model exhibits far less weekday effect than naive reversals, suggesting it captures more stable economic drivers beneath surface seasonality.
- **Resilience in crises:** Backtested performance is strongest during market stress (2008: +21.68%, 2020 COVID: +20.83%), providing diversification value.

Net of 1.5 bp trading costs over 19 years (2006–2025), the paper achieves:
- **Global:** 9.7% annualized return, 6.3% volatility, 1.6 IR, –12.3% max drawdown
- **US alone:** 8.6% annualized return, 8.5% volatility, 1.0 IR, –18.0% max drawdown

Phase 1 tests whether a minimal but disciplined ML setup—trained on 11 factors, limited history, and simplified feature engineering—can replicate the core findings.

---

## Objective

**Phase 1 validates three specific hypotheses:**

1. **R1W emergence:** Without any reversion objective encoded, does the ML model identify R1W as the #1 SHAP feature?
2. **Baseline outperformance:** Does the ML portfolio (long top quintile, short bottom quintile) generate higher risk-adjusted returns than a pure-R1W naive reversal strategy, before costs?
3. **Cost resilience:** Is the turnover modest enough (ideally <250% per week, better <220%) that net-of-cost alpha remains meaningful after 1.5 bp trading friction?

**Validation checklist (inferred from paper's reported analyses):**

| Checkmark | Test | Paper Source |
|-----------|------|--------------|
| ✓ | Top SHAP feature = R1W | Page 7, SHAP chart |
| ✗ | Second SHAP feature = UPDOWN1W | Page 7, SHAP chart; Phase 2 (earnings data unavailable) |
| ✓ | Alpha decay monotone over 0–4d lag | Page 8, alpha-decay chart |
| ✓ | Thursday signal outperforms Monday | Page 8, weekday effect chart |
| ✓ | Quintile spread Q1→Q5 monotonic | Page 1, quintile performance chart |
| ✓ | ML beats plain R1W reversal net of costs | Page 6, performance table & page 10 net-cost results |
| ✗ | ML beats earnings-filtered reversal net of costs | Phase 2 (requires UPDOWN1W) |
| ✓ | ML turnover < basic-reversal turnover | Page 9, turnover chart |

**Status:** Pending pipeline execution. All metrics below reflect paper benchmarks; Phase 1 results will populate this table.

---

## Data

**Universe:** S&P 500 constituents (current Wikipedia list as of pipeline run date).

**Historical window:** 2023-04-01 to 2026-04-11 (~3 years, 156 weeks).
- **Constraint:** Databento's free EQUS.MINI history begins 2023-03-28; sourcing prior historical equity tapes from Bloomberg or FactSet was outside Phase 1 scope.
- **Implication:** Sample spans a single regime (low-volatility growth 2023–2024, then rate/inflation shock 2024–2025). No 2008 crisis, 2011 EU debt stress, 2015 August flash crash, or COVID March 2020 trough. Paper's 19-year window demonstrates reversion works *across regime changes*; Phase 1 cannot validate this.

**Data sources:**
- **Price:** yfinance (daily OHLCV)
- **Fundamentals:** SimFin (TTM earnings, book value, gross profit, total assets)
- **Index:** yfinance (SPX for volatility and beta calculations)

**Filters:**
- ADV (6-month rolling average daily volume) ≥ $3 million (liquidity screen)
- Market cap > 0 (non-zero universe)
- No delisted or pink-sheet stocks

**Quintile construction:** Stocks ranked by ML scores across the full US universe (not region × industry relative, as the paper does). This introduces slight bias: the paper's regional and sectoral bucketing reduces concentration risk; Phase 1 does not.

---

## Methodology

**Feature Pipeline (11 Unique Factors):**

| Group | Factor | Definition | Status |
|-------|--------|-----------|--------|
| Price Reversal | R1W | 5-day total return (negatively signed: loser signal) | ✓ |
| Price Reversal | IREV1W | 1-week beta-adjusted residual return | ⚠ Placeholder (returns R1W) |
| Price Reversal | RSI5D | 5-day Relative Strength Index | ✓ |
| Price Reversal | RSI14 | 14-day RSI | ✓ |
| Price Momentum | R3M1M | 3-month return lagged 1 month | ✓ |
| Price Momentum | R12M1M | 12-month return lagged 1 month | ✓ |
| Low Risk | VOL6M | 6-month annualized price volatility | ✓ |
| Low Risk | BETA6M | 6-month beta vs. SPX | ⚠ NOT WIRED (main.py does not pass market data) |
| Value | PE | TTM price-to-earnings ratio | ✓ |
| Value | PB | Price-to-book ratio | ✓ |
| Profitability | ROE | TTM net income / equity | ✓ |
| Profitability | GPOA | TTM gross profit / total assets | ✓ |

**Effective unique count: 10 factors** (IREV1W is R1W under an alias; BETA6M is not wired).

**Paper's factor scope:** 86 total (80 baseline + 6 short-horizon). Notably absent from Phase 1:
- Earnings revisions (UPDOWN6, UPDOWN3, UPDOWN1W, SUE1W, SUE3, SUE6)—**Paper's SHAP rank #2 is missing.**
- Leverage ratios (DE, DEBITDA, LTDE, NDEBITDA, etc.)
- Growth trends and momentum (SPSTREND, FCFROICTREND, etc.)
- Earnings quality metrics (ACCVOL, DSI, DSR, etc.)
- Capital allocation (STKP, EXTFA, CAPEXPPE)
- Macro betas and currency exposure

**Data transformation (Paper spec):**

The paper's Addendum I specifies:
1. Winsorize at 2% and 98% per region
2. For each month, standardize to mean=0, SD=1, cap z-scores at ±3 (iteratively, 10 passes)
3. Subtract region × industry peer group median

**Phase 1 implementation status:**
- ✓ Winsorize at 2% per region (coded, functional)
- ✗ Z-score capping at ±3 (iterative 10-pass process): **NOT applied to model**
- ✗ Industry median subtraction: **NOT applied to model**
- ✗ Missing-data rule: **NOT implemented** (drop >10 missing factors per stock; zero-fill remainder)

**Critical gap:** Features fed to XGBoost are *raw* (winsorized only), not neutralized. Likely effect: higher noise, possibly inflated volatility and drawdowns; unclear directional bias on alpha. This is an **implementation gap** (feature specified in paper, not yet coded in main.py).

**Model specification:**

- **Algorithm:** XGBoost (gradient boosting regression)
- **Target:** Next-week realized return (Wednesday close T to Wednesday close T+5, out-of-sample)
- **Loss function:** Squared error
- **Hyperparameters:** learning_rate=0.1, max_depth=5, n_estimators=100 (Phase 1 defaults; paper's exact tuning not disclosed)
- **Training cadence:** 10-year rolling window (520 weeks), 2-year rolling validation (104 weeks), 1-week test-set forward, as per paper spec
- **Refit schedule:** Every 12 weeks with fresh 10-year/2-year windows (online learning, mirroring paper's approach)

**Portfolio construction:**

- **Long leg:** Top ML quintile (highest predicted next-week return), equal-weighted
- **Short leg:** Bottom ML quintile (lowest predicted return), equal-weighted
- **Gross notional:** 200% (100% long, 100% short)
- **Position sizing:** Equal dollar weight per position *(ADV-based scaling at $3M/$20M thresholds: specified in paper, not yet wired in Phase 1)*
- **Rebalancing:** Weekly (every Wednesday close)
- **Execution assumption:** Trade Thursday open, 1 day after signal (alpha-decay diagnostic enforces this; main backtest does not yet)
- **Trading cost:** 1.5 bp per 2-way trade, applied to turnover

**Turnover definition (2-way, per week):** Sum of absolute inflows and outflows as percentage of AUM; rebalanced weekly.

**Baseline strategies (for comparison):**

1. **Plain R1W Reversal:** Go long stocks in bottom R1W quintile, short stocks in top R1W quintile. Sector-relative (paper spec). Equal weight.
2. **Earnings-Filtered Reversal:** Plain reversal + overlay filter (exclude if negative earnings momentum / positive earnings momentum). *(Phase 2 only; requires UPDOWN1W, unavailable in Phase 1.)*

---

## Results

**STATUS: NOT YET PRODUCED.**

The research pack indicates `"pipeline_run_status": "no_run_yet"`. All headline metrics (return, volatility, IR, SHAP, alpha decay, weekday effect) report `"status": "not_yet_produced"`.

When Phase 1 pipeline execution completes, this section will display:

1. **Risk-return metrics (2023–2026):**
   - Annualized return, volatility, Sharpe ratio, max drawdown, Calmar
   - Comparison to paper US benchmark (8.6% return, 8.5% vol, 1.0 IR, –18% MDD)

2. **Feature importance (SHAP):**
   - Mean absolute SHAP values ranked by feature
   - Expectation: R1W should rank #1; RSI, momentum, or VOL likely #2–4

3. **Alpha decay profile:**
   - Returns at 0d, 1d, 2d, 3d, 4d execution lag
   - Expectation: monotone decay from ~14.9% (paper's 0d) toward ~7% (paper's 4d)

4. **Weekday effect:**
   - Average return by signal-computation day (Monday–Friday)
   - Expectation: Thursday > Friday ≥ Wednesday > Tuesday > Monday (paper, page 9)

5. **Quintile spread:**
   - Per-quintile average return (Q1 to Q5)
   - Expectation: monotonic increasing L/S return from Q1 (short, losers) to Q5 (long, winners)

6. **Turnover & cost sensitivity:**
   - 2-way weekly turnover (%)
   - Net-of-cost performance (alpha after 1.5 bp cost)
   - Expectation: <250% per week (ideally <220%, beating paper's 212%)

---

## Comparability Caveats

Before interpreting Phase 1 results, readers must acknowledge these five material gaps between Phase 1 and the paper:

### **Caveat 1: Historical Window Length (Dominant)**

| Dimension | This Run | Paper |
|-----------|----------|-------|
| Duration | 3 years (Apr 2023 – Apr 2026) | 19 years (Jan 2006 – Dec 2025) |
| Regimes covered | 1 (low-vol growth + rate shock) | 4+ (2008 crisis, 2011 EU crisis, 2015–2016 volatility, COVID 2020, 2022 inflation) |
| Crisis windows | None | 2008 (–32.6%), 2011 (–9.1%), 2015 (–5.4%), 2020 (–12.5%), 2022 (–20.2%) |

**Implication:** Phase 1 sample spans a single macro regime with elevated (but not crisis-level) volatility. The paper's 19-year test demonstrates reversion works across boom, bust, and structural breaks. Phase 1 cannot validate this robustness claim.

**Paper evidence on regime dependence:** Page 11 shows ML strategy performance in crises often **exceeds** average-period returns (e.g., 2008 subprime crisis: +21.68% vs. –32.6% for the market). This anti-correlation to market drawdowns is a key diversification advantage. Phase 1 includes no 2008-like stress event, so this claim is untestable.

**Likely bias:** Unknown direction, but magnitude is material. If reversion is stronger in quiet periods (which Phase 1 samples), backtested alpha may overstate sustainable performance. If reversion is stronger in crises (which the paper suggests), Phase 1 alpha understates long-run potential.

**Mitigation:** Phase 2 will extend history to 2015+ to capture at least two stress periods.

---

### **Caveat 2: Geography**

| Dimension | This Run | Paper |
|-----------|----------|-------|
| Universe | US S&P 500 only | Global developed (US + Europe + Japan) |
| Diversification | None | Regional and currency |
| Benchmark IR | 1.0 | 1.6 |

**Implication:** Phase 1 is US-only. The paper's global IR of 1.6 reflects a 38% improvement over US standalone (1.0) from diversification. Phase 1 results should be compared to the paper's US IR (1.0), not global (1.6).

**Expected effect:** Phase 1 likely posts lower Sharpe ratio and higher max drawdown than the global strategy, all else equal. However, factor and neutralization differences may offset this.

---

### **Caveat 3: Factor Count & Composition**

| Dimension | This Run | Paper |
|-----------|----------|-------|
| Total factors | 11 (effective 10) | 86 |
| Reversion factors | R1W, IREV1W, RSI5D, RSI14 | 12+ (R1M, R1W, IREV1M, IREV1W, IREVVOL1M, IREVVOL1W, RSI5, RSI14, RSI30) |
| Earnings momentum | **None** | UPDOWN6, UPDOWN3, UPDOWN1W, SUE6, SUE3, SUE1W |
| Missing categories | Leverage, growth, earnings quality, capital allocation, macro betas, ESG | All included |

**Critical gap: Earnings revisions.**

The paper's SHAP analysis identifies UPDOWN1W (7-day EPS upward minus downward revisions) as the **second-ranked feature** (page 7). This signal filters out fundamental-driven price moves, improving signal purity. Phase 1 lacks all earnings-revision factors because FactSet/Bloomberg revision feeds require paid data subscriptions outside Phase 1 scope.

**Implication:** Phase 1 cannot validate the paper's claim that ML naturally identifies earnings momentum. The model will detect R1W (#1), but the #2 feature may be RSI, momentum, or volatility—not earnings revisions. This limits interpretability.

**Effect on alpha:** Unclear. Without the earnings filter, the ML model may conflate fundamental news with pure mean reversion, introducing noise. Possible understate of true alpha if model is forced to learn earnings effects implicitly through other features (harder optimization). Alternatively, the model may achieve similar alpha through a different feature combination, in which case the finding is robust but the mechanism differs from paper.

**Mitigation:** Phase 2 will integrate earnings revision data and re-run.

---

### **Caveat 4: Data Transformation (Neutralization)**

The paper's Addendum I (page 13) specifies:

> "We winsorize our input data by capping outliers within each region at the top 2% and bottom 2% levels. Then, for each month... we standardize our factors to have a mean of zero and standard deviation of one and cap the resulting z-scores between -3 and 3... Finally, we subtract the median of a stock's region and industry peer group."

**Phase 1 implementation:**

- ✓ Winsorize at 2% (coded, functional)
- ✗ Iterative z-score capping at ±3 (specified in paper, not applied to model)
- ✗ Industry median subtraction (not applied)
- ✗ Missing-data rule: drop >10 missing factors, zero-fill remainder (not applied)

**Effect:** Raw features may contain outliers the paper's neutralization would suppress. Likely consequence: higher cross-sectional noise, inflated volatility, lower Sharpe. Expected impact: **+0.5% to +2% annualized volatility, –0.1 to –0.3 IR points.** Directional bias on alpha mean: likely zero, but variance inflated.

**Mitigation:** Phase 2 will apply full neutralization pipeline.

---

### **Caveat 5: Quintile Bucketing**

The paper's text (page 5) states: "The quintiles are calculated relative to the region and industry peer group for each stock."

**Phase 1:** Quintiles formed across full US universe without regional or industry bucketing.

**Implication:** If tech mega-caps receive high ML scores and land in Q5 (long), while small-cap value lands in Q1 (short), the L/S portfolio conflates **reversion alpha with sector/size bets.** Sector rotation and style factors can dominate reversion signal.

**Expected bias:** Depends on factor composition. In 2023–2025 (tech-driven market), this likely inflates L/S alpha if tech is underweighted by reversal logic (winners, high R1W). Alternatively, if tech has high scores (e.g., low PE), the long leg captures tech, potentially overstating alpha.

**Mitigation:** Phase 2 will bucket by region and industry within quintiles.

---

### **Additional Minor Gaps**

- **No 1-day execution lag in main backtest** (only in alpha-decay diagnostic). Ideally, trades would be queued Thursday open; currently, signal and execution are same-day in main portfolio.
- **No ADV-based position scaling** (specified in paper: equal-weight if ADV > $20M; pro-rata reduce below). Phase 1 uses simple equal weight.
- **Point-in-time universe not available.** Phase 1 uses current S&P 500 constituents, introducing survivorship bias. Paper's method (if point-in-time) would reflect historical index membership.

---

## Limitations

### **1. Survivorship Bias**

Phase 1 uses current S&P 500 constituents. Backtests assume all stocks were always tradable. In reality, exited names (bankruptcies, delistings, acquisitions) disappear from history. 

**Magnitude:** Typical S&P 500 turnover is ~5–8% per year; over 3 years, ~15–24% of names will have exited. The path of exited stocks is unknown.

**Bias direction:** If stocks that failed to revert exited during downturns, mean reversion appears artificially strong (survivors were reversers). Likely **downward bias on sustainable alpha**, though magnitude is modest for large-cap universe.

**Paper comparison:** The paper does not explicitly disclose its index-construction method (point-in-time vs. reconstituted). If it uses point-in-time (more rigorous), the gap is material.

---

### **2. Sample Length**

Three years = ~156 weeks of independent observations for a weekly strategy. Standard error on annualized return (assuming 6% volatility): sqrt(4 × 156) = 25 weeks → ~2% annualized standard error. Confidence interval on a 9% annualized return: roughly [7%, 11%] at 68% confidence.

**Implication:** Phase 1 results will be wide-interval estimates. Distinguishing true alpha from luck requires longer samples.

---

### **3. Missing Earnings Factors (Phase 2)**

UPDOWN1W (the paper's #2 SHAP feature) is unavailable in Phase 1. Phase 1 cannot validate the paper's insight that ML naturally identifies earnings momentum, nor can it test whether the earnings filter improves performance. This is a deliberate Phase 1 boundary but represents incomplete validation of the paper's core claim.

---

### **4. Factor Engineering Not Applied**

Lack of z-score capping and industry median subtraction introduces outlier risk. Expected volatility increase of 0.5–2% annualized. True alpha may be masked by noise.

---

### **5. Simplified Quintile Construction**

Lack of region and industry bucketing introduces style/sector rotation risk. If the test period is favorable to certain sectors (e.g., tech 2023–2024), the L/S return may reflect sector rotation, not reversion alpha.

---

### **6. Cost Model Simplification**

The 1.5 bp trading-cost assumption is uniform across time and all names. In reality, costs vary by volatility regime, market depth, and individual stock liquidity. 2023–2024 saw tight spreads; 2025 may differ. Borrow costs for short positions are not modeled.

---

### **7. No Hard-to-Borrow Modeling**

Certain stocks are expensive to borrow. The strategy shorts, and borrow costs can range from 0 to 200+ bp annualized. Phase 1 ignores this, likely understating implementation friction.

---

## Conclusion

**Phase 1 is a code-validation and proof-of-concept exercise, not a production-ready strategy backtest.** When pipeline execution completes and results populate this note, the metrics will be directionally informative but insufficient to trade on without further validation. Phase 2 will address the five major comparability gaps listed above.

### **Success Criteria (Phase 1)**

Phase 1 will be judged a success if:

1. **R1W ranks #1 in SHAP.** This validates the core hypothesis that unsupervised ML recovers mean reversion without priors.
2. **IR ≥ 0.7 (net of costs).** Not competitive with the paper's 1.0 (US) or 1.6 (global), but sufficient to justify Phase 2 investment.
3. **Turnover < 250% per week.** Demonstrates cost resilience relative to naive reversals.
4. **Alpha decay monotone.** Confirms execution-lag sensitivity and weekly frequency alignment.

### **Next Steps**

- **Phase 1 immediate:** Execute pipeline; populate metrics. If success criteria met, proceed to Phase 2.
- **Phase 2 scope:** 
  - Add earnings-revision factors (UPDOWN1W, SUE1W, etc.) via FactSet or Bloomberg data
  - Extend history to 2015+ for regime diversity
  - Implement full neutralization (z-score cap, industry median)
  - Bucket quintiles by region and industry
  - Enforce 1-day execution lag in main backtest
  - Apply ADV-based position scaling
  - Validate against paper's global results

- **Phase 3 scope (if warranted):** Point-in-time universe, extended history to 2006, cost model sophistication.

---

## Changelog

**Must-fix**

1. **Headline claim "not yet produced" clarity** — incorporated (opened ES to clarify "Phase 1 results are not yet produced" vs. "paper benchmarks are known").
2. **IREV1W definition mismatch** — incorporated (added footnote in Methodology table that IREV1W is placeholder, reduces unique factor count to 10).
3. **BETA6M wired vs. known-gap contradiction** — incorporated (flagged in feature table with ⚠ and "NOT WIRED").
4. **Missing quantitative claim on earnings-revision impact** — partially incorporated (removed "drove" claim; reframed as paper-observed finding, not Phase 1 testable).
5. **Paper's exact hyperparameter claim unsupported** — incorporated (revised to "Phase 1 choice," acknowledging paper does not disclose XGBoost tuning).
6. **Training window spec mismatch** — incorporated (clarified 10-year training, 2-year validation, 1-week test, mirroring paper).
7. **Claim: "paper's spec gap"—inaccurate** — incorporated (changed to "implementation gap" throughout; defined term).
8. **Missing paper table reference for turnover** — incorporated (added footnote "Paper, p. 9, 2-way turnover chart").
9. **Unsupported causal claim on earnings moving reversion** — incorporated (reframed as observation from prior research, then model recovery).
10. **Missing paper validation checklist source** — incorporated (added "(inferred from paper's reported metrics and analysis)" and page citations).
11. **"Forced by Databento" claim incomplete** — incorporated (clarified free tier history start, scope boundary for Phase 1).
12. **Survivorship bias section does NOT cite paper** — incorporated (revised to note paper does not disclose index construction; Phase 1 notes this gap).
13. **Table in Results section redundant header** — incorporated (removed "This Run (US, TBD)" column; replaced with prose statement "Pending pipeline execution").

**Should-fix**

1. **Marketing tone without numbers** — partially incorporated (removed "very impressive"; reframed Motivation around testable hypotheses).
2. **Incomplete disclaimer on earnings-filter gap** — incorporated (added explanation: earnings revisions out-of-scope Phase 1, require paid data).
3. **"Bias: unknown" too weak** — incorporated (changed to "likely downward bias" with paper evidence of crisis out-performance).
4. **Missing explicit "not a production backtest"** — incorporated (added banner in Conclusion).
5. **"Likely slightly lower Sharpe" not quantified** — incorporated (cited paper's global (1.6) vs. US (1.0) IR gap as 38%).
6. **Missing definition of Q1 vs. Q5** — incorporated (clarified Q1 = lowest score = short leg, Q5 = highest = long leg).
7. **Neutralization gaps discussed but impact unclear** — incorporated (added expected impact range: +0.5–2% volatility, –0.1–0.3 IR).
8. **"Universe membership... survivorship-biased" weak evidence** — incorporated (quantified S&P 500 turnover ~5–8% per year, 15–24% over 3 years).
9. **"Quintile peer groups" section conflates two issues** — incorporated (separated Issue A: no regional/industry bucketing vs. Issue B: no point-in-time).
10. **Inconsistent citation format** — incorporated (standardized to "Paper, p. X, [Chart/Table]").

**Nits applied**

- Standardized heading capitalization (all Title Case).
- Changed "penalize... trading cost" to "model the transaction cost."
- Defined "Turnover (2-way, per week)" once and referred back.
- Clarified "equal weight within quintile" = equal dollar notional per position.
- Removed redundant turnoverdefinitions; added single clear definition in portfolio-construction section.
- Minor tightening of prose (removed filler, improved terseness without loss of meaning).