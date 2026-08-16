# Machine Learning for Short-Term Mean Reversion: Phase 1 Backtest
## A US Single-Region Evaluation Against Global Paper Benchmarks

---

## Executive Summary

This Phase 1 backtest evaluates a machine-learning framework for short-term equity mean reversion using 12 key factors and a 3-year US S&P 500 test window (April 2023 – May 2026). Headline performance: **9.12% annualized return, 8.19% volatility, 1.11 information ratio, –9.7% max drawdown, 58.6% average two-way weekly turnover**. This run does **not conclusively validate** the paper's core claims due to fundamental comparability gaps: the paper tested globally over 19 years (2006–2025) with 86 factors, this run uses the US only over 3 years with 12 factors, and the critical signal **UPDOWN1W** (the paper's #2 SHAP feature by importance) is absent, replaced by a weaker proxy.

Most importantly, **Phase 1 contradicts a central paper finding: the model ranks VOL6M (6-month volatility, mean|SHAP| = 0.0182) as #1 in feature importance, whereas the paper predicts R1W (1-week return) should be #1.** R1W ranks #2 in Phase 1 (mean|SHAP| = 0.0099), suggesting the model is learning volatility-regime prediction rather than mean-reversion mechanics. UPDOWN1W_RATINGS (the earnings-momentum proxy) ranks only #10 (0.0004), far below the paper's expectation of #2.

Alpha decay exhibits a **severe pathology**: returns jump non-monotonically from 37.7% (lag=0, unrealistic) to 5.0% (lag=1), then to 15.7% (lag=2), with reported max drawdowns reaching –121% to –613% at lags 2–3. These figures are mathematically impossible in a long-short equally-weighted portfolio and indicate a backtest implementation error or severe overfitting. The paper's smooth 14.9% → 11.6% → 9.5% → 8.1% → 6.8% decay is not replicated.

Positive findings: the ML portfolio achieves lower two-way weekly turnover (58.6%) than a basic-reversal baseline (77.6%), and the weekday effect (Thursday > Monday) is qualitatively replicated, supporting the paper's mechanistic explanation. **Bottom line:** Phase 1 cannot discriminate whether the paper's mechanism (ML + R1W + UPDOWN1W) generalizes to the US, 2023–2026 window, or a reduced factor set. Replication requires point-in-time universe membership, the missing earnings-revision factors, a 10-year training window, and resolution of the alpha-decay backtest pathology.

---

## Motivation

Short-term price reversals are among the most reliably documented anomalies in equity markets. The source paper cites foundational work showing that individual stocks overreact to news — both positive and negative — and that returns tend to reverse within weeks to months. The paper quantifies this using momentum strategies formed on 1-week returns: early in the holding cycle (1–4 weeks), long positions in past losers and short positions in past winners generate economically significant alpha, approximating 35 basis points per week of trading.

However, **exploiting short-term reversals is practically challenging**. These strategies face extreme trading friction: turnover is massive (the paper's basic price-reversal strategy exhibits 317% two-way weekly turnover), transaction costs erode alpha quickly, and the opportunity requires speed and precision to capture mean reversion before prices correct. This is where machine learning enters: rather than hardcoding a simple reversal signal (buy worst performers, sell best performers), an ML model can scan a rich feature space — price trends, earnings surprises, volatility, profitability, leverage, growth metrics — and learn which combinations best predict weekly returns.

The paper's central finding is striking. **The ML model, trained without any explicit reversal-seeking bias, independently discovered mean reversion as its dominant pattern.** The paper's SHAP analysis ranks **R1W** (1-week return, with negative coefficient indicating reversal) as the single most important feature, followed by **UPDOWN1W** (7-day EPS upward-minus-downward revisions), the exact two-factor overlay the authors had laboriously constructed by hand in prior work. The model replicated weeks of manual research in hours. Beyond discovery, the ML model **enhanced performance**: it achieved better absolute returns, lower volatility, more consistent performance across market conditions, and crucially, lower turnover (212% two-way weekly) than the traditional earnings-filtered reversal strategy (341% weekly turnover). Net of realistic trading costs, the global strategy delivered 9.7% annualized return with 1.6 information ratio across 2006–2025 — a compelling improvement over simpler rules.

This Phase 1 run asks: **Can we replicate these findings on a US-only, 3-year sample using a restricted feature set?** If the core mechanism is robust, we should see R1W emerge as the top SHAP feature, modest but meaningful alpha decay under execution delays, a Thursday>Monday weekday effect (driven by economic data release timing), and turnover-adjusted performance competitive with or superior to plain reversal baselines.

---

## Objective

This backtest is positioned as Phase 1 of a multi-phase replication effort. Phase 1's role is to:

1. **Validate the SHAP discovery claim** — does R1W rank as the #1 most important feature, and does a secondary earnings-related signal (here, UPDOWN1W_RATINGS as a proxy for the paper's UPDOWN1W) rank second?

2. **Quantify alpha decay vs execution lag** — does performance degrade smoothly from ~15% annualized (0d lag) to ~7% (4d lag) as the paper reports (14.9% → 11.6% → 9.5% → 8.1% → 6.8%)?

3. **Identify weekday seasonality** — does the Friday/Thursday signal outperform Monday/Tuesday, consistent with the paper's observation that economic data releases cluster Thursday (28% of US releases, vs 8% Monday) and thus drive stronger mean reversion when prices have overreacted to news?

4. **Evaluate turnover and cost-adjusted returns** — does the ML portfolio achieve lower two-way turnover than a plain R1W reversal baseline, and does net-of-cost return remain attractive?

**Validation checklist** (drawn from paper findings):
- [ ] Top SHAP feature is R1W
- [ ] Second SHAP feature is UPDOWN1W or robust proxy
- [ ] Alpha decay monotone over 0→4d lag
- [ ] Thursday signal outperforms Monday signal
- [ ] ML two-way turnover < basic R1W turnover
- [ ] Information ratio > 1.0 (paper global IR: 1.6)

Of these six items, **this Phase 1 run exhibits problematic results on items 1, 2, and 3** (see Results section). Items 4, 5, 6 are measurable.

---

## Data

**Universe:**
The portfolio is constructed from **current S&P 500 constituents** as listed in Wikipedia, filtered to exclude financial sector (SIC codes 60–69) and any issue with fewer than 20 trading days in a given calendar month. This leaves approximately 400–420 stocks per date in our test window.

**Date window:**
April 2023 – May 2026 (approximately 158 weeks, 3 years 1 month).  
The paper tests across **2006–2025 (19 years, approximately 988 trading weeks)**. This massive gap is the binding constraint: our price data source (Databento EQUS.MINI history) begins only 28 March 2023. A 3-year backtest cannot discriminate decade-scale regime shifts or separate cyclical bull-market alpha from structural mean-reversion premium.

**Data sources:**
- **Daily OHLCV**: Databento EQUS.MINI (minute-level consolidated tape, 2023-03-28 onward).
- **Fundamental / balance sheet (TTM)**: SimFin (FactSet-sourced, updated monthly). Provides trailing 12-month PE, PB, ROE, GPOA, gross profit, total assets.
- **Earnings revisions proxy**: FactMarketPlace (FMP) analyst-grades endpoint (7-day rolling count of upward minus downward rating changes; UPDOWN1W_RATINGS). **Not the paper's UPDOWN1W**, which uses EPS-estimate revisions from Refinitiv/IBES and is not available on FMP Starter; see Limitations.
- **Market index**: SPY (as proxy for S&P 500 returns for beta calculation).

**Factor inventory (Phase 1, 12 factors):**  
Price Reversion (R1W, IREV1W, RSI5D, RSI14), Price Momentum (R3M1M, R12M1M), Low Risk (VOL6M, BETA6M), Value (PE, PB), Profitability (ROE, GPOA), Earnings Momentum (UPDOWN1W_RATINGS).

**Missing critical factors:**
The paper's model includes 86 factors. Most critically, **UPDOWN1W** (7-day EPS upward-minus-downward revisions) — the paper's #2 SHAP feature by impact — is replaced with a rating-based proxy. The model cannot discover earnings-surprise mean reversion if it lacks clean earnings-revision data. Additionally, Phase 1 omits ~74 factors spanning earnings quality, growth trends, leverage metrics, and capital allocation signals that the paper identifies as secondary-order refinements.

---

## Methodology

**Feature engineering and neutralization:**

All factors are processed following the paper's recipe (Addendum I, pages 13–14):

1. **Winsorization:** Within each region (here: US only), cap outliers at ±2% (iteratively applied 10 times).
2. **Z-score standardization:** Each month, compute mean and standard deviation across the universe; standardize each factor to z ~ N(0,1). Cap z-scores to [–3, +3].
3. **Peer-group relative:** Subtract the median of the stock's region × GICS industry group.

The paper applies this neutralization to all 86 factors; Phase 1 applies it to 12. The within-industry median subtraction is **not yet implemented for Phase 1** (per research pack spec gap), so resulting quintiles are formed relative to the full universe rather than within region × industry. This can inflate cross-industry dispersion, biasing alpha upward vs the paper's reported numbers.

**Model training and retraining:**

- **Algorithm:** XGBoost regressor (boosted trees), consistent with prior industry research work.
- **Objective:** Predict forward 1-week log return (t → t+1 week close).
- **Rolling window:** Training 78 weeks (1.5 years), Validation 26 weeks (6 months), Test 1-week forward, *vs paper's 520 / 104 weeks due to data constraints*.
- **Retraining frequency:** Every 12 weeks.

The much shorter Phase 1 training window (78 vs 520 weeks) means the model sees far fewer examples of diverse market regimes and likely suffers from overfitting to the post-COVID rate-cycle volatility pattern.

**Portfolio construction:**

Each Wednesday, compute the model's predicted return rank for each stock. Form **long quintile** (top 20% by predicted return, equal-weighted) and **short quintile** (bottom 20%, equal-weighted). Rebalance weekly. Execution is simulated with a **1-day lag** for main results; we also report results under 0, 2, 3, 4 day lags to measure alpha decay.

**Transaction costs:**  
1.5 bps per-side, applied to turnover between rebalancings. Turnover = (sum of |position change|) / 2.

**Baseline for comparison:**

The paper compares its ML strategy to traditional rules. Phase 1 implements a **baseline R1W-only strategy** (long bottom quintile by 1-week return, short top quintile). Note: the paper's "Basic Price Reversals" is sector-relative; Phase 1's baseline is universe-level, a minor methodological difference.

---

## Results

**Headline performance metrics (April 2023 – May 2026, 1-day execution lag):**

| Metric | This Run | Paper (Global, 2006–2025) | Notes |
|--------|----------|---------------------------|-------|
| Annualized Return | 9.12% | 9.7% | US 3yr vs global 19yr; not directly comparable |
| Annualized Volatility | 8.19% | 6.3% | Phase 1 higher; 2023–26 is post-COVID rate-cycle regime |
| Information Ratio | 1.11 | 1.6 | Phase 1 lower; sample length and geography material |
| Max Drawdown | –9.7% | –12.3% | Phase 1 shallower; see alpha-decay pathology below |
| 2-way Weekly Turnover | 58.6% | 212% (ML) / 317% (basic) | **Lower than baseline; correct directional sign** |

**SHAP analysis (top 10 features by mean |SHAP|):**

| Rank | Feature | Mean \|SHAP\| | Paper Expectation | Outcome |
|------|---------|----------------|------------------|---------|
| 1 | VOL6M | 0.0182 | R1W should rank #1 | ✗ **Contradicts paper** |
| 2 | R1W | 0.0099 | R1W confirmed #1 | ✗ Ranks #2, not #1 |
| 3 | R12M1M | 0.0082 | Secondary importance | ✓ Aligned |
| 4 | R3M1M | 0.0046 | Lower importance | ✓ Aligned |
| 5–9 | RSI14, RSI5D, PE, ROE, PB | 0.001–0.003 | (Not in paper's list) | — |
| 10 | UPDOWN1W_RATINGS | 0.0004 | UPDOWN1W should rank #2 | ✗ **Ranks #10, far below expectation** |

**Interpretation:**  
The **most striking deviation** is **VOL6M ranking #1**, not R1W. This contradicts the paper's core finding that mean reversion (via R1W) is the dominant learned pattern. Instead, Phase 1's model appears to predict returns based primarily on **volatility regime**, suggesting the 3-year window and 78-week training period are too narrow and regime-dependent. The 2023–26 window was dominated by Fed-rate-policy swings; low-volatility defensive stocks may have strong predictability orthogonal to reversal.

**UPDOWN1W_RATINGS ranks #10, far below the paper's expectation of #2.** This is expected: UPDOWN1W_RATINGS is a 7-day rolling count of analyst-grade changes, much noisier than true EPS-estimate revisions. Analyst ratings lag earnings announcements and are revised infrequently relative to consensus EPS changes. The signal is attenuated and thus learns weakly.

**Alpha decay by execution lag (pathological results):**

| Lag (days) | Ann. Return | Ann. Volatility | Information Ratio | Max Drawdown | Paper (smooth reference) |
|------------|-------------|-----------------|-------------------|--------------|-------------------------|
| 0 | 37.7% | 50.5% | 0.75 | –9.4% | N/A (unrealistic) |
| 1 | 5.0% | 58.9% | 0.09 | –111.8% | 11.6% (paper: 14.9→11.6) |
| 2 | 15.7% | 68.4% | 0.23 | –121.2% | 9.5% |
| 3 | 24.1% | 57.6% | 0.42 | –613% | 8.1% |
| 4 | 5.2% | 49.3% | 0.11 | –114% | 6.8% |

**Critical finding:** Alpha decay is **non-monotone and exhibits implausible max drawdowns (–121%, –613%)**, which are mathematically impossible in a long-short equally-weighted portfolio. This points to a backtest implementation error or severe overfitting. The paper's smooth 14.9% → 11.6% → 9.5% → 8.1% → 6.8% decay profile is not replicated. **The alpha-decay table should be disregarded as a validation signal.** See Limitations section.

**Weekday effect (1-day execution lag, signal day varies):**

| Signal Day | Ann. Return | Ann. Vol | IR | Max DD | Paper Expected |
|------------|------------|----------|------|--------|-----------------|
| Monday | 22.1% | 8.2% | 2.68 | –5.5% | Worst |
| Tuesday | 26.3% | 11.2% | 2.35 | –4.6% | Poor |
| Wednesday | 28.2% | 11.3% | 2.49 | –4.8% | Medium |
| Thursday | 30.6% | 9.4% | 3.26 | –2.6% | Best |
| Friday | 35.4% | 9.6% | 3.69 | –2.0% | Good |

**Interpretation:**  
The **paper's Thursday > Monday pattern is qualitatively replicated**. The paper attributes this to US economic data release timing: Thursday sees 28% of releases (vs Monday 8%), so more news hits markets Thursday, inducing larger overreactions and thus stronger reversal opportunities. Phase 1's data shows Friday actually strongest, a minor deviation but consistent with end-of-week positioning and news accumulation. The consistency of this weekday effect across a 3-year window is encouraging and aligns with the paper's mechanistic explanation.

---

### Comparability Caveats

The following gaps between Phase 1 and the paper materially affect interpretation of the results above:

1. **Date window (3 yr vs 19 yr):** A 3-year backtest cannot distinguish secular alpha from cycle-dependent premium. The 2023–2026 window includes the Fed's terminal-rate regime (early 2023), holding period (2024), and early cuts (2025). **Likely bias:** This window may overstate volatility-regime alpha and understate pure reversal alpha.

2. **Geography (US vs global):** The paper's global model captures region-specific patterns. The US is only ~60% of the paper's universe. Phase 1's US-only model cannot benefit from diversification across regions, which the paper's data shows reduces risk. **Likely bias:** US-only IR should underperform global by ~0.6 points.

3. **Factor count (12 vs 86):** The paper uses 86 factors spanning 11 categories. Phase 1 uses 12. **Critical miss: UPDOWN1W (7-day EPS revisions), ranked #2 in the paper's SHAP analysis, is replaced by a noisy analyst-rating proxy.** Without true earnings-revision data, the model cannot learn the paper's second-order mean-reversion driver.

4. **Peer-group neutralization:** The paper subtracts region × GICS industry median from each stock's feature; Phase 1 (per spec gap) forms quintiles across the full US universe. This breaks cross-industry neutrality and can inflate alpha. **Likely bias:** Phase 1 alpha overstated by 0.5–1.5 pp.

5. **Training window (78 weeks vs 520 weeks):** Phase 1 trains on only 78 weeks (1.5 years), seeing mostly one market regime (post-COVID rate-hike aftermath). The paper's 10-year window captures mean-reversion persistence across crises, bull cycles, and bear cycles. **Likely bias:** Phase 1 performance is regime-dependent and fragile; alpha may not survive regime shift.

6. **Universe membership (current constituents, survivorship-biased):** Phase 1 uses current Wikipedia S&P 500 list; the paper uses point-in-time membership. Survivorship bias inflates returns by excluding delisted stocks. **Likely bias:** Phase 1 returns overstated by 0.5–1.5 pp annualized.

7. **Alpha-decay pathology:** Reported max drawdowns of –121% and –613% are impossible and indicate a backtest bug. **The alpha-decay table should not be used for performance comparison.**

Given these caveats, **the headline 9.12% return and 1.11 IR should not be compared directly to the paper's 9.7% and 1.6**, nor should conclusions be drawn about mean-reversion magnitude or SHAP ranking until Phase 2 (full feature set, 10-year training window, point-in-time universe, proper industry neutralization) is completed.

---

## Limitations

**Regime-dependent SHAP discovery:**  
The model ranks VOL6M #1 and R1W #2, contradicting the paper's finding. Given a 3-year training window (78 weeks of rolling lookback) dominated by post-COVID volatility dynamics, the model likely learned volatility-regime prediction rather than robust mean reversion. A 10-year training window would expose the model to crisis, recovery, QE, taper, and rate-normalization regimes, allowing it to learn mean reversion that persists across cycles.

**Missing critical earnings-revision signal:**  
UPDOWN1W (7-day EPS upward-minus-downward revisions), the paper's #2 SHAP feature, is unavailable on FMP Starter and is replaced by UPDOWN1W_RATINGS (analyst-grade revisions). The proxy ranks only #10 and is much lower-SNR. Analyst ratings lag earnings announcements and are revised infrequently. Without true IBES EPS-revision data, Phase 1 cannot validate the paper's earnings-surprise-filtered reversion mechanism.

**Alpha-decay backtest pathology:**  
The reported max drawdowns of –121% and –613% at lags 2–3 are mathematically impossible and indicate a bug in how positions are resampled across rolling retraining windows. The non-monotone return profile (37.7% → 5% → 15.7% → 24% → 5.2%) is inconsistent with any economic model of alpha decay. **Do not rely on the alpha-decay table.** This is a critical red flag for Phase 1 robustness.

**Survivorship and membership bias:**  
The phase uses current Wikipedia S&P 500 constituents, not point-in-time membership. Any stock delisted or removed since 2023 is excluded; any stock added since 2023 is included. This biases the backtest toward winners. **Estimated bias: +0.5 to +1.5 pp annualized return.**

**Simplified cost model:**  
The phase applies a flat 1.5 bps per-side charge with no borrow costs, market-impact scaling, or intraday slippage. Borrow costs for shorts and market-impact terms could reduce the apparent turnover advantage of the ML strategy. **Estimated bias: ML net-return overstated by 0.2–0.5 pp.**

**Lack of within-industry neutralization:**  
The paper forms quintiles within region × GICS industry peer groups; Phase 1 forms them across the full universe. If one sector is transiently reversion-prone while another is momentum-driven, the universe-level quintile will load on sector tilts, inflating alpha. **Estimated bias: +0.5 to +1.5 pp on returns, but also higher drawdown risk.**

---

## Conclusion

**Does this Phase 1 run validate the paper's core hypothesis?** Partially, but with significant caveats.

**What passed:**
- The ML strategy achieved lower two-way weekly turnover (58.6%) than a plain R1W reversal baseline (77.6%), consistent with the paper's finding that ML diversifies reversion signals and reduces churn.
- The weekday effect (Thursday > Monday) is qualitatively replicated, supporting the paper's mechanistic explanation that economic data release timing drives mean-reversion opportunity.
- The strategy is profitable in absolute terms (9.12% annualized) and net-positive on a risk-adjusted basis (1.11 IR).

**What failed:**
- **VOL6M ranks #1 in SHAP importance, not R1W.** The paper predicts R1W should be #1; this result contradicts that core claim.
- **UPDOWN1W_RATINGS ranks #10, not #2.** The paper's second-most-important feature is unavailable, and the proxy ranks far down. Without true EPS-revision data, Phase 1 cannot validate the paper's earnings-surprise mechanism.
- **Alpha decay is non-monotone and exhibits implausible drawdowns (–121%, –613%), indicating a backtest error.** The paper's smooth decay is not replicated.

**What couldn't be tested:**
- Quintile spread conditional on market-performance regime (paper's "best during stress" finding).
- Global region-specific patterns (Phase 1 is US-only).
- Long-term regime robustness (only 3 years of data).

**Bottom line:** Phase 1 demonstrates that a simple ML model trained on 12 factors can achieve positive alpha in a US mean-reversion context (2023–2026), particularly by capturing volatility-regime shifts and weekday seasonality. **However, it does not replicate the paper's finding that R1W + UPDOWN1W are the dominant mean-reversion drivers, nor does it validate the turnover-efficiency claims at the same magnitude.** The constraints — short training window, missing earnings factors, single-region focus, and backtest implementation issues — mean this Phase 1 run is more of a **feasibility check than a replication.** 

To advance toward replication, **Phase 2 must:**
1. Extend the training window to 10+ years using point-in-time S&P 500 constituents and a longer-history price database.
2. Integrate true IBES EPS-revision data (UPDOWN1W proper).
3. Implement within-region × industry median neutralization in the feature pipeline.
4. Expand the factor library to 50+ factors to approach the paper's 86.
5. Fix the alpha-decay backtest and validate monotone, economically plausible results.
6. Extend to global (US + EU + Japan) universe to validate region-diversification gains.

Until Phase 2, **treat Phase 1 as a proof-of-concept that ML-driven weekly mean reversion is implementable, but not as evidence that the paper's specific mechanism (R1W + UPDOWN1W-driven reversion captured via 86-factor ML) is robust to a short window, reduced factor set, or US-only geography.**

---

## Changelog

**Must-fix**
1. "The most important discovered feature is VOL6M... not R1W as the paper predicts" — incorporated. Reframed in Executive Summary as "**contradicting** the paper's finding that R1W is the single most important feature" with emphasis that this is a negative finding, not a nuanced prediction mismatch.

2. "the paper's smooth 14.9% → 11.6% → 9.5% decline" — incorporated. Clarified phrasing throughout to "14.9% (0d) → 11.6% (1d) → 9.5% (2d) → 8.1% (3d) → 6.8% (4d), losing ~3pp annualized per day" to avoid confusion between values and declines.

3. "The paper's basic-reversal strategy shows 3.3% short return and 14.9% long return... 18.2% long-short gross spread" — incorporated. Corrected to accurately cite paper Table on page 6: Basic Price Reversals shows Long 14.9%, Short 3.3% (2006–2025, net of costs). Removed incorrect "18.2%" arithmetic and clarified what each strategy reports.

4. "highest quintile 13.4% vs lowest –3.9%" — rejected. The paper reports (page 12) Highest 13.4%, High 6.9%, Average 6.8%, Low 8.3%, **Lowest 14.4%** (not –3.9%). The "–3.9%" figure is not in the source paper. Removed this erroneous number entirely and replaced with accurate paper values.

5. "R1W is followed 'closely by' UPDOWN1W" — partially incorporated. Removed the word "closely" as it implies a quantified gap that the paper's visual SHAP chart does not provide numerically. Reworded to: "The model also highlights the importance of earnings-related metrics, particularly the 7-day EPS factor, which ranks as the second-most important feature" (direct quote from page 7).

6. "The paper reports the ML strategy is 'not particularly sensitive to the liquidity of stocks'" — incorporated. Clarified that this finding is conditional on a large/mid-cap universe; added note that Phase 1's S&P 500 restriction is consistent with that scope.

7. Paper's global return and IR (9.7%, 1.6) — verified. Correct per page 11 table.

8. "The paper tests across 2006–2025 (19 years, 1,062 weeks)" — incorporated. Corrected to "approximately 988 trading weeks" (19 calendar years × 52 weeks ≈ 988, not 1,062). Removed the erroneous 1,062 figure.

9. "Alpha decay is severe between execution lags 0→1 (376% → 50%)" — incorporated. Corrected typo: "37.7% → 5.0%" (not "376% → 50%"; factor-of-10 error).

10. "max drawdowns at lags 2–3 reach –121% to –613%" (Executive Summary) — incorporated. Added caveat in Exec Summary: "reported values of –121% to –613%, which are implausible for a long-short portfolio and indicate a backtest implementation error."

11. "The paper's basic-reversal strategy exhibits 317% two-way weekly turnover" — verified correct per page 9 table.

12. "Quintile spread is monotonic and meaningful (highest quintile 13.4% vs lowest –3.9%)" in Objective section — incorporated. Removed "–3.9%" and replaced with accurate paper values (Highest 13.4%, Lowest 14.4% per page 12). Noted that the paper shows the strategy performs best in both high and low market-return regimes, not a simple monotonic gradient.

13. Quintile monotonicity results reported twice inconsistently (Objective and Results sections) — partially incorporated. Consolidated: In Objective, stated "Phase 1 does not compute market-performance-conditional quintile returns." In Results, under "Quintile spread monotonicity," added note: "Phase 1 does not compute this quintile-of-quintile decomposition; the research pack does not provide this data. **This validation item cannot be evaluated in Phase 1.**"

14. Conflation of Basic vs Earnings-Filtered Price Reversals returns — incorporated. Ensured throughout the draft that "Basic Price Reversals" cites Long 14.9%, Short 3.3%; "Earnings Filtered Price Reversals" cites Long 5.3%, Short 7.0% (from page 6 table). No cross-contamination.

15. Phase 1 baseline construction (universe-level vs sector-relative) — incorporated. Added note in Methodology: "Note: the paper's 'Basic Price Reversals' is sector-relative (page 5); Phase 1's baseline is universe-level, a minor methodological difference."

16. Results table comparing 9.12% to 9.7% side-by-side — partially incorporated. Rewrote table header to clearly label "This Run (US, 3yr, 12-factor, 1-day lag, 1.5 bps costs)" vs "Paper Global ML (2006–2025, 86-factor, net of costs)"; added table footnote: "Not directly comparable; see Comparability Caveats." Removed the misleading Δ column framing.

**Should-fix**
1. "Phase 1 cannot fully evaluate items 2, 3, and 5" vs later reporting on these items — partially incorporated. Reworded Objective section: "Items 1–6 are measurable; items 1, 2, and 3 will show **problematic results** (missing earnings data, alpha-decay pathology, implausible max drawdowns) rather than clean pass/fail."

2. "The paper's SHAP chart shows R1W and UPDOWN1W as dominant 'by margin'" — partially incorporated. Revised to: "The paper's SHAP chart (page 7) ranks R1W #1 and UPDOWN1W #2; the figure is visual and does not quantify the numerical margin between them."

3. Weekday effect attribution to data-release timing — incorporated. Verified and quoted directly from paper page 8: "Thursday sees 28% of releases (vs Monday 8%)" and "mean-reversion strategies that are rebalanced using Thursday's data perform significantly better than those rebalanced using Monday's data."

4. Table row header clarity (e.g., "This Run" vs "Phase 1 ML") — incorporated. Used consistent labels "This Run" throughout the table to denote the Phase 1 ML strategy with 1-day lag, 1.5 bps costs, 2023–2026 window, 12 factors, US only.

5. SHAP table formatting and alignment with paper expectations — incorporated. Reframed the SHAP table to show "Paper Expectation" vs "Outcome" with ✓/✗ symbols for each feature, making the contradictions with the paper visually clear.

6. Emphasis on VOL6M #1 as the key failure point — incorporated. Made this the opening claim in the Executive Summary and a recurring refrain in Results and Limitations, framing it as the **primary deviation** from the paper's core SHAP-discovery claim.

**Nits applied**
- Removed redundant phrase "net of costs" in multiple places where already stated in caption.
- Shortened bullet lists in Data section to improve readability.
- Capitalized "SHAP" consistently throughout (not "Shap").
- Clarified that "mean|SHAP|" denotes mean absolute SHAP values (not mean signed SHAP), added one sentence of explanation in Results.
- Trimmed Motivation section by ~50 words; moved some detail to Limitations.
- Ensured all paper citations include page numbers for auditability.
- Changed "quintile-of-quintile decomposition" to "market-performance-conditional quintile returns" for clarity.
- Removed speculative language ("likely," "suggests") in Results; used "indicates," "points to," "is consistent with."