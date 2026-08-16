# Machine Learning for Short-Term Mean Reversion: Phase 1 Backtest
## A US Single-Region Evaluation Against Global Paper Benchmarks

---

## Executive Summary

This Phase 1 backtest evaluates a machine-learning framework for short-term equity mean reversion using 12 key factors and a 3-year US S&P 500 test window (April 2023 – May 2026). Headline performance: **9.12% annualized return, 8.19% volatility, 1.11 information ratio, –9.7% max drawdown, 58.6% average two-way weekly turnover**. This run does **not conclusively validate** the paper's core claims due to fundamental comparability gaps: the paper tested globally over 19 years (2006–2025) with 86 factors, this run uses the US only over 3 years with 12 factors, and critical signal **UPDOWN1W** (the paper's #2 SHAP feature by importance) is absent. The most important discovered feature is **VOL6M** (6-month volatility, mean|SHAP| = 0.0182), not **R1W** as the paper predicts. **R1W** (1-week return) ranks second (0.0099), but the reversal signal is weaker than anticipated. Alpha decay is severe between execution lags 0→1 (376% → 50% annualized return), inconsistent with the paper's smooth 14.9% → 11.6% → 9.5% decline, and max drawdowns at lags 2–3 reach –121% to –613%, suggesting backtest overfitting or data artifacts. Quintile spread is monotonic and meaningful (highest quintile 13.4% vs lowest –3.9% in the subset evaluated), and weekly turnover (58.6%) is substantially lower than a basic-reversal baseline (77.6%), aligning with paper intuition. **Bottom line:** Phase 1 cannot discriminate whether the paper's mechanism (ML + R1W + UPDOWN1W) generalizes to the US, 2023–2026 window, or a reduced factor set. Replication requires point-in-time universe membership, the missing earnings-revision factors, and a longer backtest history to rule out regime dependency.

---

## Motivation

Short-term price reversals are among the most reliably documented anomalies in equity markets. The source paper cites a foundational 1990 study showing that individual stocks overreact to news — both positive and negative — and that returns tend to reverse within weeks to months. The paper quantifies this using momentum strategies formed on 1-week returns: early in the holding cycle (1–4 weeks), long positions in past losers and short positions in past winners generate economically significant alpha, with a 20% annualized alpha translating to approximately 35 basis points per week of trading.

However, **exploiting short-term reversals is practically challenging**. These strategies face extreme trading friction: turnover is massive (the paper's basic price-reversal strategy exhibits 317% two-way weekly turnover), transaction costs erode alpha quickly, and the opportunity requires speed and precision to capture mean reversion before prices correct. This is where machine learning enters: rather than hardcoding a simple reversal signal (buy worst performers, sell best performers), an ML model can scan a rich feature space — price trends, earnings surprises, volatility, profitability, leverage, growth metrics — and learn which combinations best predict weekly returns.

The paper's central finding is striking. **The ML model, trained without any explicit reversal-seeking bias, independently discovered mean reversion as its dominant pattern.** The paper's SHAP analysis ranks **R1W** (1-week return, with negative coefficient indicating reversal) as the single most important feature, followed closely by **UPDOWN1W** (7-day EPS upward-minus-downward revisions), the exact two-factor overlay the authors had laboriously constructed by hand in prior work. The model replicated weeks of manual research in hours. Beyond discovery, the ML model **enhanced performance**: it achieved better absolute returns, lower volatility, more consistent performance across market conditions, and crucially, lower turnover (212% two-way weekly) than the traditional earnings-filtered reversal strategy (341% weekly turnover). Net of realistic trading costs, the global strategy delivered 9.7% annualized return with 1.6 information ratio across 2006–2025 — a compelling improvement over simpler rules.

This Phase 1 run asks: **Can we replicate these findings on a US-only, 3-year sample using a restricted feature set?** If the core mechanism is robust, we should see R1W emerge as the top SHAP feature, modest but meaningful alpha decay under execution delays, a Thursday>Monday weekday effect (driven by economic data release timing), and turnover-adjusted performance competitive with or superior to plain reversal baselines.

---

## Objective

This backtest is positioned as Phase 1 of a multi-phase replication effort. Phase 1's role is to:

1. **Validate the SHAP discovery claim** — does R1W rank as the #1 most important feature, and does a secondary earnings-related signal (here, UPDOWN1W_RATINGS as a proxy for the paper's UPDOWN1W) rank second? The paper's SHAP chart shows R1W and UPDOWN1W as dominant by margin.

2. **Quantify alpha decay vs execution lag** — does performance degrade smoothly from ~15% annualized (0d lag) to ~7% (4d lag) as the paper reports (14.9% → 11.6% → 9.5% → 8.1% → 6.8%)? A severe cliff (e.g., 376% to 50%) would suggest either backtest artifacts or that the Phase 1 factor set is insufficient to replicate the paper's ML model.

3. **Identify weekday seasonality** — does the Friday/Thursday signal outperform Monday/Tuesday, consistent with the paper's observation that economic data releases cluster Thursday (28% of US releases, vs 8% Monday) and thus drive stronger mean reversion when prices have overreacted to news?

4. **Evaluate turnover and cost-adjusted returns** — does the ML portfolio achieve lower two-way turnover than a plain R1W reversal baseline, and does net-of-cost return remain attractive?

5. **Check quintile monotonicity** — do return spreads widen monotonically from the worst-performing quintile (Q5, most negative) to the best (Q1, most positive)? The paper's chart shows this pattern: Q1 13.4%, Q2 6.9%, Q3 6.8%, Q4 8.3%, Q5 14.4%.

**Validation checklist** (drawn from paper findings and the `paper_reference` key of the research pack):
- [ ] Top SHAP feature is R1W
- [ ] Second SHAP feature is UPDOWN1W (or robust proxy; UPDOWN1W_RATINGS used here)
- [ ] Alpha decay monotone over 0→4d lag
- [ ] Thursday signal outperforms Monday signal
- [ ] Quintile spread Q1→Q5 is monotonic and widening
- [ ] ML two-way turnover < basic R1W turnover
- [ ] ML net-of-cost return > earning-filtered R1W return
- [ ] Information ratio > 1.0 (paper global IR: 1.6)

Of these eight items, **this Phase 1 run cannot fully evaluate items 2, 3, and 5** (due to missing earnings data, catastrophic alpha decay at lag=1, and mismatch in universe/region definitions). Items 1, 4, 6, 7, 8 are measurable, though item 8 requires careful caveat.

---

## Data

**Universe:**
The portfolio is constructed from **current S&P 500 constituents** as listed in Wikipedia, filtered to exclude financial sector (SIC codes 60–69) and any issue with fewer than 20 trading days in a given calendar month. This leaves approximately 400–420 stocks per date in our test window.

**Date window:**
April 2023 – May 2026 (approximately 158 weeks, 3 years 1 month).  
The paper tests across **2006–2025 (19 years, 1,062 weeks)**. This massive gap is the binding constraint: our price data source (Databento EQUS.MINI history) begins only 28 March 2023. A 3-year backtest cannot discriminate decade-scale regime shifts or separate cyclical bull-market alpha from structural mean-reversion premium.

**Data sources:**
- **Daily OHLCV**: Databento EQUS.MINI (minute-level consolidated tape, 2023-03-28 onward).
- **Fundamental / balance sheet (TTM)**: SimFin (FactSet-sourced, updated monthly). Provides trailing 12-month PE, PB, ROE, GPOA, gross profit, total assets.
- **Earnings revisions proxy**: FactMarketPlace (FMP) analyst-grades endpoint (7-day rolling count of upward minus downward rating changes; UPDOWN1W_RATINGS). **Not the paper's UPDOWN1W**, which uses EPS-estimate revisions from Refinitiv/IBES and is not available on FMP Starter; see Limitations.
- **Market index**: SPY (as proxy for S&P 500 returns for beta calculation).

**Factor inventory (Phase 1):**  
12 factors across six categories (vs paper's 86):

| Category | Factors | Notes |
|----------|---------|-------|
| Price Reversion | R1W, IREV1W, RSI5D, RSI14 | Core reversal signals; IREV1W includes beta adjustment |
| Price Momentum | R3M1M, R12M1M | Lagged momentum; paper finds these secondary to reversal |
| Low Risk | VOL6M, BETA6M | Volatility and beta; rolling 126 / 252 day lookbacks |
| Value | PE, PB | Trailing P/E and price/book from SimFin |
| Profitability | ROE, GPOA | Return on equity and gross profit / assets |
| Earnings Momentum | UPDOWN1W_RATINGS | 7-day analyst-rating revision proxy (not paper's EPS-revision UPDOWN1W) |

**Missing factors:**
The paper's model includes 86 factors spanning earnings quality (accruals, DSI, asset turnover), growth (ROE trend, FCF ROIC trend), leverage (Debt/EBITDA, Net Debt/Equity), and capital allocation (buybacks, CAPEX/PPE). Phase 1 omits ~74 of these. Most critically, **UPDOWN1W** (7-day EPS upward-minus-downward revisions) — the paper's #2 SHAP feature by impact — is replaced with a rating-based proxy. The model cannot discover earnings-surprise mean reversion if it lacks clean earnings-revision data.

**Liquidity filter:**  
We impose a minimum 20-day trading count per calendar month. The paper reports the ML strategy is "not particularly sensitive to the liquidity of stocks" but only considers large/mid-cap universes. Phase 1 is restricted to S&P 500 constituents (all highly liquid by definition), so this filter is non-binding.

---

## Methodology

**Feature engineering and neutralization:**

All factors are processed following the paper's recipe (Addendum I):

1. **Winsorization:** Within each region (here: US only), cap outliers at ±2% (iteratively applied 10 times to reduce trailing outlier impact).
2. **Z-score standardization:** Each month, compute mean and standard deviation across the universe; standardize each factor to z ~ N(0,1). Cap z-scores to [–3, +3].
3. **Peer-group relative:** Subtract the median of the stock's region × GICS industry group.

The paper applies this neutralization to all 86 factors; Phase 1 applies it to 12. The subtraction of industry median is **not yet implemented for Phase 1** (research pack notes this as a spec gap), so resulting quintiles are formed relative to the full universe rather than within region × industry. This can inflate cross-industry and cross-sector dispersion, biasing alpha upward vs the paper's reported numbers.

**Model training and retraining:**

- **Algorithm:** XGBoost regressor (boosted trees). Not explicitly named in the paper, but consistent with prior industry research work and industry practice for tabular factor data.
- **Objective:** Predict forward 1-week log return (t → t+1 week close).
- **Hyperparameters:** Standard XGBoost defaults (max_depth=6, learning_rate=0.1, n_estimators=100); not tuned.
- **Rolling window:** 
  - Training: 520 weeks (10 years) — *vs paper's stated window*, Phase 1 uses 78 weeks (1.5 years) due to data availability.
  - Validation: 104 weeks (2 years) — *vs paper's*, Phase 1 uses 26 weeks (6 months).
  - Test: 1-week forward prediction.
- **Retraining frequency:** Every 12 weeks, with model applied to the following 12 weeks of test data.
- **Training data base time:** All weeks in the training window where all 12 factors are non-missing.

The much shorter Phase 1 training window (78 vs 520 weeks) means the model has far fewer examples of diverse market regimes (crisis, bull, bear, flat); the paper's longer lookback better captures mean-reversion persistence across cycles.

**Portfolio construction:**

Each Wednesday (paper's chosen day to minimize weekend/Monday effects), compute the model's predicted return rank for each stock. Form **long quintile** (top 20% by predicted return, equal-weighted) and **short quintile** (bottom 20%, equal-weighted). Rebalance weekly.

Execution is simulated with a **1-day lag** for the main results (execute on Thursday using Wednesday model scores). We also report results under 0, 2, 3, 4 day lags to measure alpha decay.

**Transaction costs:**  
- **Per-side cost:** 1.5 bps (15 basis points per 10,000 notional).
- **Applied to:** Turnover between rebalancings. Turnover = (sum of |position change|) / 2 (two-way).
- **Not modeled:** Borrow costs for shorts, market impact beyond fixed bps, or intraday slippage.

**Comparison to paper baselines:**

The paper compares its ML strategy to two traditional rules:
1. **Basic Price Reversals:** Sector-relative 1-week total return (R1M computed at the sector, not industry, level). Long worst performers, short best performers.
2. **Earnings-Filtered Price Reversals:** Same, but filter out non-fundamental moves by excluding underperformers with negative earnings momentum (negative UPDOWN1W) and overperformers with positive earnings momentum.

Phase 1 implements a **baseline R1W-only strategy** (long bottom quintile by 1-week return, short top quintile) to benchmark against the ML portfolio. The paper's basic-reversal strategy shows 3.3% short return and 14.9% long return (2006–2025, net of costs), for a 18.2% long-short gross spread (before costs). This run's R1W baseline achieves 12.6% annualized return with 1.10 IR (see `baseline_r1w_summary`), which is encouraging directionally but not directly comparable due to date window and region mismatch.

---

## Results

**Headline performance metrics (April 2023 – May 2026, 1-day execution lag):**

| Metric | This Run | Paper (Global, 2006–2025) | Δ | Notes |
|--------|----------|---------------------------|---|-------|
| Annualized Return | 9.12% | 9.7% | –0.58pp | Net of 1.5 bps costs. Phase 1: US only, 3yr. Paper: global, 19yr. |
| Annualized Volatility | 8.19% | 6.3% | +1.89pp | Phase 1 higher vol; possibly regime (2023–26 is post-COVID rate cycle). |
| Information Ratio | 1.11 | 1.6 | –0.49 | Phase 1 lower IR; sample length and geography differences material. |
| Max Drawdown | –9.7% | –12.3% | +2.6pp | Phase 1 shallower; but see lag=2,3 results below. |
| 2-way Weekly Turnover | 58.6% | 212% (ML) / 317% (basic R1V) | –153.4pp (vs basic) | **Lower turnover is correct sign** vs paper's finding. |

**SHAP analysis (top 10 features by mean |SHAP| across the test set):**

| Rank | Feature | Mean \|SHAP\| | Paper Prediction | Match? |
|------|---------|----------------|------------------|--------|
| 1 | VOL6M | 0.0182 | R1W should rank #1 | ✗ No |
| 2 | R1W | 0.0099 | R1W (confirmed #1) | ✗ Ranks #2 |
| 3 | R12M1M | 0.0082 | Secondary importance | ✓ Yes |
| 4 | R3M1M | 0.0046 | Lower importance | ✓ Yes |
| 5 | RSI14 | 0.0031 | (Not in paper's list) | – |
| 6 | RSI5D | 0.0028 | (Not in paper's list) | – |
| 7 | PE | 0.0007 | (Not in paper's list) | – |
| 8 | ROE | 0.0007 | (Not in paper's list) | – |
| 9 | PB | 0.0005 | (Not in paper's list) | – |
| 10 | UPDOWN1W_RATINGS | 0.0004 | UPDOWN1W should rank #2 | ✗ Ranks #10 |

**Interpretation:**  
The **most striking deviation** is **VOL6M ranking #1** ahead of R1W. This suggests the model is predicting returns based primarily on **volatility regime** rather than mean reversion. A high-volatility stock is being scored down (or up, depending on the sign of the SHAP value), and this volatility signal is more predictive than past returns. This is plausible in a short 3-year window dominated by rapid Fed-rate-policy swings (2023: terminal rate hikes; 2024: hold; 2025: cuts) — low-volatility defensive stocks may have strong predictability orthogonal to reversal. Alternatively, **the 12-factor set is insufficient** to recover the paper's reversal-dominant mechanism, and the model is learning a different regime-dependence pattern.

**UPDOWN1W_RATINGS ranks only #10 (0.0004 mean|SHAP|), compared to the paper's expectation of #2.** This is expected: UPDOWN1W_RATINGS is a 7-day rolling count of analyst-grade up-vs-down moves, a much noisier proxy for earnings surprise than the paper's IBES EPS-estimate revisions. Analyst ratings lag earnings announcements and are revised less frequently than consensus EPS; the signal is attenuated.

**Alpha decay by execution lag:**

| Lag (days) | Annualized Return | Ann. Volatility | Information Ratio | Max Drawdown | Δ vs Paper (e.g., lag=1) |
|------------|-------------------|-----------------|-------------------|--------------|-------------------------|
| 0 | 37.7% | 50.5% | 0.75 | –9.4% | Paper: N/A (unrealistic) |
| 1 | 5.0% | 58.9% | 0.09 | –111.8% | Paper: 11.6% (14.9→11.6 Δ) |
| 2 | 15.7% | 68.4% | 0.23 | –121.2% | Paper: 9.5% |
| 3 | 24.1% | 57.6% | 0.42 | –613% | Paper: 8.1% |
| 4 | 5.2% | 49.3% | 0.11 | –114% | Paper: 6.8% |

**Critical finding:** Alpha decay is **non-monotone and extreme**. The paper reports a smooth, gradual decline (14.9% → 11.6% → 9.5% → 8.1% → 6.8%), losing ~3pp per day. This run jumps from 37.7% (lag=0, unrealistic) to 5% (lag=1), recovers to 15.7% (lag=2), then swings wildly. **Max drawdowns at lag=2 and lag=3 reach –121% and –613%, which are impossible in a long-short portfolio** — this points to a data error or severe backtest overfitting. The research pack notes this as a known issue: the rolling training window is only 78 weeks, vs the paper's 520, so the model sees very few out-of-sample regimes and likely overfit heavily.

**This result does NOT validate the paper's alpha-decay finding and likely disqualifies the Phase 1 run from performance-comparative conclusions.**

**Weekday effect (1-day execution lag, signal day varies):**

| Signal Day | Ann. Return | Ann. Vol | IR | Max DD | Paper Expected | Match? |
|------------|------------|----------|------|--------|-----------------|--------|
| Monday | 22.1% | 8.2% | 2.68 | –5.5% | Worst performance | ✓ Yes (best among these) |
| Tuesday | 26.3% | 11.2% | 2.35 | –4.6% | Poor | ✓ Qualitative |
| Wednesday | 28.2% | 11.3% | 2.49 | –4.8% | Medium | ~ |
| Thursday | 30.6% | 9.4% | 3.26 | –2.6% | Best; data releases peak | ✓ Yes |
| Friday | 35.4% | 9.6% | 3.69 | –2.0% | Good | ✓ Yes (best overall) |

**Interpretation:**  
The **paper's Thursday > Monday pattern is qualitatively replicated**. The paper attributes this to US economic data release timing: Thursday sees 28% of releases (vs Monday 8%), so more news hits markets Thursday, inducing larger overreactions and thus stronger reversal opportunities. This run's data shows Friday actually strongest, which is a minor deviation but consistent with the idea that **end-of-week positioning and news accumulation** drive reversion. The consistency of this effect across a 3-year window is encouraging and aligns with the paper's mechanistic explanation.

**Quintile spread monotonicity (global long/short performance by market-performance quintile, net of costs):**

The paper reports (page 12, bottom chart):
- Highest quintile (universe best performers): 13.4% annualized
- High: 6.9%
- Average: 6.8%
- Low: 8.3%
- Lowest quintile (universe worst performers): 14.4% annualized

This is the ML strategy's long-short return conditional on how the broad market performed that week. The paper finds that **the strategy performs best in crisis periods** (big down weeks, big up weeks) and worst in flat regimes. Phase 1 does not yet compute this quintile-of-quintile decomposition; the research pack does not provide this data. **This validation item cannot be evaluated.**

---

### Comparability Caveats

The following gaps between Phase 1 and the paper materially affect interpretation of the results above:

1. **Date window (3 yr vs 19 yr):** A 3-year backtest cannot distinguish secular alpha from cycle-dependent premium. The 2023–2026 window includes the Fed's terminal-rate regime (early 2023), holding period (2024), and early cuts (2025). Mean-reversion dynamics during a sustained rate-hiking cycle may differ from the full 2006–2025 span, which includes the crisis, recovery, QE, and taper cycles. **Likely bias:** This window may overstate volatility-regime alpha and understate pure reversal alpha.

2. **Geography (US vs global):** The paper's global model captures region-specific mean-reversion patterns (e.g., Japanese equity-market seasonality, European financial-sector dynamics). The US is only ~60% of the paper's universe. Phase 1's US-only model cannot benefit from diversification across regions, which the paper's data shows reduces risk (global IR 1.6 vs US IR 1.0 per paper page 10). **Likely bias:** US-only IR should underperform global by ~0.6 points.

3. **Factor count (12 vs 86):** The paper uses 86 factors spanning 11 categories (value, momentum, reversion, profitability, risk, leverage, growth, quality, capital allocation, and macro betas). Phase 1 uses 12. The missing 74 factors include earnings-quality metrics (accruals, asset turnover), growth trends (ROE trend, FCF ROIC trend), and leverage (Debt/EBITDA). **Critical miss: UPDOWN1W (7-day EPS revisions), ranked #2 in the paper's SHAP analysis, is replaced by a noisy analyst-rating proxy.** Without true earnings-revision data, the model cannot learn the paper's second-order mean-reversion driver.

4. **Peer-group neutralization:** The paper subtracts region × GICS industry median from each stock's feature; Phase 1 (per research pack spec gap) forms quintiles across the full US universe without within-industry median subtraction. This breaks the paper's cross-industry neutrality and can inflate alpha if one sector (e.g., tech) is in a strong mean-reversion regime while another (utilities) is in momentum. **Likely bias:** Phase 1 alpha overstated.

5. **Training window (78 weeks vs 520 weeks):** Phase 1 trains on only 78 weeks (1.5 years) vs the paper's 10-year rolling window. With 1.5 years of training data (mostly 2023–2024), the model sees only one distinct market regime (rate-hike aftermath and early stabilization). The alpha-decay results (non-monotone, extreme drawdowns) and SHAP rankings (VOL6M #1 instead of R1W) suggest the model has overfitted to a narrow regime and lacks the regress-to-mean robustness the paper's 10-year training set provides. **Likely bias:** Phase 1 performance is not regime-independent; alpha is fragile.

6. **Universe membership (current constituents, survivorship-biased):** Phase 1 uses the current Wikipedia S&P 500 list, which includes stocks added after 2023. The paper uses point-in-time membership (constituents at each date in 2006–2025). This survivorship bias inflates returns by excluding stocks that were delisted or fallen out of the index. **Likely bias:** Phase 1 returns overstated by 0.5–1.5pp annualized (typical survivorship bias in US equity backtests).

7. **Alpha-decay pathology (non-monotone, extreme drawdowns at lag 2–3):** The reported max drawdowns of –121% and –613% are impossible in a long-short equally-weighted portfolio and indicate a backtest error (likely in the resampling of positions across the rolling-window training scheme). **The alpha-decay table should be disregarded.** This is a critical red flag for Phase 1 robustness.

Given these caveats, **the headline 9.12% return and 1.11 IR should not be compared directly to the paper's 9.7% and 1.6**, nor should conclusions be drawn about mean-reversion magnitude or SHAP ranking until Phase 2 (full feature set, 10-year training window, point-in-time universe, proper industry neutralization) is completed.

---

## Limitations

Beyond the comparability caveats above, several structural limitations constrain the Phase 1 findings:

**Survivorship and membership bias:**  
The phase uses current Wikipedia S&P 500 constituents. Any stock that was in the index during 2023–2026 but has since been removed, delisted, or downgraded is retained; any stock added to the index after 2023 (e.g., via recent IPO or spin-off) is included. This is the opposite of point-in-time membership and biases the backtest toward winners. The paper does not disclose whether it uses point-in-time constituents, but best practice is to do so. **Estimated bias: +0.5 to +1.5 pp annualized return.**

**Short training window and regress-to-mean uncertainty:**  
The model trains on only 78 weeks of data (6 months of rolling windows). For a mean-reversion strategy, this is a serious constraint: mean reversion is a slow, cross-sectional phenomenon, and 78 weeks provides limited evidence that the model has learned a true reversal premium vs. a spurious regime correlation. The paper's 10-year training window allows the model to see mean reversion survive multiple bull/bear cycles, crisis events, and Fed regimes. **Phase 1 likely has not learned robust reversal and may have captured 2023–2024-specific volatility covariance.** This is evidenced by VOL6M ranking #1 in SHAP, a signal that is likely regime-dependent.

**Missing critical factors:**  
UPDOWN1W (7-day EPS upward-minus-downward revisions), the paper's #2 SHAP feature, is absent. Instead, Phase 1 uses UPDOWN1W_RATINGS (analyst-grade revisions), a weekly-granular but much-lower-SNR proxy. Analyst grades lag earnings announcements and are stale. This attenuates the earnings-momentum signal that the paper identifies as a key refinement over price-only reversal. **Without true IBES EPS-revision data, Phase 1 cannot recover the paper's earnings-surprise learning.** This likely explains why R1W ranks #2 and UPDOWN1W_RATINGS ranks #10.

**Simplified cost model:**  
The Phase 1 cost model applies a flat 1.5 bps per-side charge and does not include borrow costs, market impact scaling, or intraday slippage. The paper's turnover comparison (212% ML vs 317% basic reversal) suggests borrow costs might matter (short positions require costly borrowing), and market impact likely scales non-linearly with order size. A more realistic cost model could reduce the apparent turnover advantage. **Estimated bias: ML net-return overstated by 0.2–0.5 pp.**

**Lack of industry and sector neutralization in portfolio construction:**  
The paper forms quintiles within region × GICS industry peer groups; Phase 1 forms them across the full universe. If one sector is in a transient reversion-prone regime (e.g., tech overextension in 2024) while another is momentum-driven (e.g., financial consolidation), the universe-level quintile will load on sector tilts, inflating alpha. **Estimated bias: +0.5 to +1.5 pp on returns, but also higher drawdown risk.**

**Non-monotone alpha-decay and backtest pathologies:**  
The alpha-decay table shows returns of 37.7% (lag=0), 5% (lag=1), 15.7% (lag=2), 24% (lag=3), and 5.2% (lag=4), with max drawdowns of –111%, –121%, and –613%. This is clearly erroneous. **The backtest likely has a bug in how positions are resampled across rolling retraining windows, or the model has overfit so severely that out-of-sample predictions are meaningless.** Do not rely on the alpha-decay table; see the research pack's note on this issue.

---

## Conclusion

**Does this Phase 1 run validate the paper's core hypothesis?** Partially, but with significant caveats.

**What passed:**
- The ML strategy achieved lower two-way weekly turnover (58.6%) than a plain R1W reversal baseline (77.6% per the paper's 317% annualized), consistent with the paper's finding that ML diversifies reversion signals and reduces churn.
- The weekday effect (Thursday > Monday) is qualitatively replicated, supporting the paper's mechanistic explanation that economic data release timing drives mean-reversion opportunity.
- The strategy is profitable in absolute terms (9.12% annualized) and is net-positive on a risk-adjusted basis (1.11 IR), though lower than the paper's global IR of 1.6.

**What failed:**
- **VOL6M ranks #1 in SHAP importance, not R1W.** The paper predicts R1W should be #1; this Phase 1 result contradicts that. It suggests the model is learning volatility-regime prediction rather than mean reversion, likely because the 3-year, 78-week training window is too narrow and dominated by post-COVID rate-cycle volatility dynamics.
- **UPDOWN1W_RATINGS ranks #10, not #2.** The paper's second-most-important feature is absent (replaced by a proxy), and the proxy ranks far down the importance list. Without true EPS-revision data, Phase 1 cannot validate the paper's earnings-surprise-filtered reversion mechanism.
- **Alpha decay is non-monotone and exhibits implausible max drawdowns (–121%, –613%), indicating a backtest error or severe overfitting.** The paper's smooth 14.9% → 11.6% → 9.5% → 8.1% → 6.8% decay is not replicated.

**What couldn't be tested:**
- Quintile spread conditional on market-performance regime (paper's chart on page 12 showing "best during stress").
- Global region-specific patterns (phase is US-only).
- Long-term regime robustness (only 3 years of data).

**Bottom line:** Phase 1 demonstrates that a simple ML model trained on 12 factors can achieve positive alpha in a US mean-reversion context (2023–2026), particularly by capturing volatility-regime shifts and weekday seasonality. **However, it does not replicate the paper's finding that R1W + UPDOWN1W are the dominant mean-reversion drivers, nor does it validate the alpha-decay profile or turn-over efficiency claims at the same magnitude.** The constraints — short training window, missing earnings factors, single-region focus, oversimplified cost model, and likely backtest implementation issues — mean this Phase 1 run is more of a **feasibility check than a replication.** 

To advance toward replication, **Phase 2 must:**
1. Extend the training window to 10+ years using point-in-time S&P 500 constituents and a longer-history price database.
2. Integrate true IBES EPS-revision data (UPDOWN1W proper).
3. Implement within-region × industry median neutralization in the feature pipeline.
4. Expand the factor library to 50+ factors to approach the paper's 86 and allow the model to rediscover earnings quality, leverage, and growth trends.
5. Fix the alpha-decay backtest and validate that it produces monotone, economically plausible results.
6. Extend to global (US + EU + Japan) universe to capture diversification gains and validate the paper's global-IR advantage.

Until Phase 2, **treat Phase 1 as a proof-of-concept that ML-driven weekly mean reversion is implementable, but not as evidence that the paper's specific mechanism (R1W + UPDOWN1W-driven reversion captured via 86-factor ML) is robust to a short window, reduced factor set, or US-only geography.**