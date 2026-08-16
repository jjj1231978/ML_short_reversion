# Research Note: ML-Driven Short-Term Equity Reversion — Phase 1 Backtest Status

## Executive Summary

**This backtest has not yet executed.** The research pack confirms configuration and implementation readiness, but all headline metrics, portfolio results, SHAP analysis, alpha decay, and weekday diagnostics remain unproduced. The planned run targets a 3-year window (Apr 2023–Apr 2026) on the S&P 500 using 12 Phase 1 factors with 78-week rolling retraining. Against the paper's US benchmark (IR 1.0, 8.6% net return, 2-way turnover 212%), this run is structurally constrained by **missing earnings revisions factors** (UPDOWN1W, SUE1W—the paper's #2 ranked SHAP feature), **unwired neutralization** (features fed raw to the model), and **universe survivorship bias** (current Wikipedia constituents vs historical membership). Once execution completes, the run should validate whether one-week price reversals (R1W) and volatility-momentum crosses drive alpha predictably, but the absence of fundamentals-based revision signals substantially limits the test's power to replicate the paper's full model performance.

---

## Motivation

Short-term equity reversion—the empirical finding that last week's losers tend to outperform this week's winners over a 1–4 day horizon—has persisted across regimes and asset classes for decades. Traditional statistical approaches (simple sign reversals, piecewise-linear momentum crossovers) capture the effect but suffer from low signal-to-noise ratios and high execution costs. The source paper (industry research, Global Style Counselling, April 2025) hypothesizes that machine learning, trained on an 86-factor library combining price history, volatility micro-structure, earnings revisions, and sector dynamics, can isolate predictive nonlinearities and interaction terms that boost the information ratio and reduce turnover relative to baseline reversals.

The paper's key empirical claims, evaluated on 19 years of global data (2006–2025), are:

1. **R1W (1-week return) ranks as the single most important feature** by SHAP value—confirming that price reversals remain the dominant driver even in a complex ML context.
2. **UPDOWN1W (7-day cumulative EPS upward minus downward revisions) ranks second**, highlighting that markets systematically underreact to near-term earnings surprises, especially when coupled with price weakness.
3. **Monotone alpha decay across 0–4 day execution lags**: The paper shows that same-day signals (lag=0) deliver an alpha of 14.9% annualized, degrading to 6.8% by lag=4, consistent with short-term market frictions and information leakage.
4. **Thursday signals outperform Monday signals by a material margin** (28% vs 8% of data releases), reflecting that Thursday reversals often precede Friday short-covering and the weekend break.
5. **Two-way weekly turnover of 212%** (rebalanced daily within the week), substantially lower than naive price-reversal baselines (317% basic, 341% earnings-filtered), proves that ML feature interactions reduce turnover.
6. **Quintile spreads remain monotonic Q1→Q5**, with no evidence of mean-reversion saturation or quant crowding at extreme decile boundaries.

This backtest aims to validate those claims on US equities over a shorter, more recent window (Apr 2023–Apr 2026), using Phase 1 factors only (earnings revisions deferred to Phase 2).

---

## Objective

This specific run is designed to answer: **Does a machine-learning reversion strategy, trained on high-frequency price and volatility features alone, reproduce the headline performance and SHAP rankings reported by the paper on a 3-year out-of-sample window?**

The validation checklist inherited from the specification document includes eight testable assertions:

| # | Assertion | Status |
|---|-----------|--------|
| 1 | Top SHAP feature = R1W | Cannot evaluate (SHAP not yet produced) |
| 2 | #2 SHAP feature = UPDOWN1W | Cannot evaluate; UPDOWN1W absent from Phase 1 |
| 3 | Alpha decay monotone over 0d→4d lag | Cannot evaluate (alpha-decay diagnostic not run) |
| 4 | Thursday signals > Monday signals | Cannot evaluate (weekday diagnostic raises NotImplementedError) |
| 5 | Quintile spread Q1→Q5 monotonic | Cannot evaluate (predictions summary not produced) |
| 6 | ML beats plain R1W baseline net of costs | Cannot evaluate (portfolio summary not produced) |
| 7 | ML beats earnings-filtered baseline | Cannot evaluate (earnings filters not in Phase 1) |
| 8 | ML 2-way turnover < 212% | Cannot evaluate (portfolio summary not produced) |

**Evaluation readiness:** Zero of eight checks can be assessed until the pipeline executes. The run configuration is technically sound—train/val split (78 weeks / 26 weeks), retrain frequency (12 weeks), cost model (1.5 bps + 1-day lag)—but structural gaps prevent full replication of the paper's findings.

---

## Data

**Universe:** S&P 500, constituents as of April 2023 via Wikipedia snapshot. This is a significant departure from the paper's methodology, which reconstructs historical S&P 500 membership using FactSet or Bloomberg. Current-snapshot constituents create **survivorship bias**: companies that exited the index between 2023 and 2026 are excluded, inflating realized returns by removing the "weight of losers" that would have been held during their decline. For a 3-year backtest, this bias likely inflates reported information ratio by 10–50 basis points.

**Date window:** 01 April 2023 through 11 April 2026 (3 years + 10 days). This window is constrained by **Databento EQUS.MINI data availability**, which begins 28 March 2023. The paper's benchmark spanned 19 years (2006–2025), a vastly longer sample capturing multiple regimes (GFC, taper tantrum, 2020 COVID crash, 2022 rate shock, 2023 mega-cap rally). A 3-year window excludes major market dislocations and is dominated by a rising-rate, then falling-rate environment with concentrated mega-cap outperformance—not necessarily representative of reversion alpha's stability across cycles.

**Source vendors:** Databento (bars + trades, 15-minute resolution) for price, volume, and volatility micro-structure. Earnings data: **absent**. The paper's UPDOWN1W and earnings-surprise (SUE) factors require a real-time earnings event feed; Phase 1 substitutes basic fundamental data only.

**Sector and ADV filters:** An ADV (average daily volume) floor of $3M is defined in the specification but **not applied** in the current main.py pipeline. The paper's position scaling rules (reduce exposure if ADV < $20M) are also absent. This means the backtest will include highly illiquid names and estimate execution costs as a flat 1.5 bps; slippage will be underestimated.

**Factor count:** 12 Phase 1 factors intended. The specification lists placeholders for:
- R1W (1-week return) — present
- IREV1W (residual 1-week return vs. market beta) — implemented as plain R1W (incorrect)
- Volatility measures: IVOL, HVOL, SKEW, KURT
- Momentum crosses: MOM_CROSS1W, MOM_CROSS1M
- Micro-structure: ROLL, SPREAD, DEPTH
- BETA6M (6-month market beta) — missing, not passed to builder

Missing vs. the paper's full 86-factor library:
- All earnings revisions (UPDOWN1W, SUE1W, SUE3, SUE6) — **Phase 2**
- Industry relative-value factors (E/P, P/B, FCF yield within peer groups) — missing
- Analyst recommendation changes and estimate revisions (APFR, AREV) — missing
- Technical factors (RSI, MACD, Bollinger Band positions) — missing

The 12-factor Phase 1 subset is roughly 14% of the paper's library, covering only price-action and volatility micro-structure. This feature thinness will limit model expressiveness and likely inflate the relative importance of R1W (since fewer competing signals are present).

---

## Methodology

**Feature pipeline:**

1. **Raw construction:** For each trading date and stock, compute:
   - R1W: cumulative return over trailing 5 trading days
   - IREV1W: R1W minus beta-adjusted market return (currently returns R1W; BETA6M missing)
   - IVOL, HVOL, SKEW, KURT: rolling (20-day or 60-day) moments of intraday returns
   - MOM_CROSS1W, MOM_CROSS1M: sign changes in momentum indicators
   - ROLL, SPREAD, DEPTH: effective bid-ask spreads and order-book depth from Databento trades
   - BETA6M: 6-month rolling beta vs. SPX close (not passed to builder in current main.py)

2. **Neutralization (specified but not wired):** The specification defines a three-step neutralization:
   - Winsorize at ±3 standard deviations
   - Iterative z-score within each industry and date
   - Subtract industry-level median
   
   **Current status:** These routines exist in `neutralize.py` but are **not called** by main.py. Features fed to the model are **raw**, retaining industry and sector exposure. This violates the paper's methodology (which explicitly applies per-region, per-industry peer-group z-scoring) and will cause the model to pick up sector rotations as "alpha," inflating in-sample fit and likely harming out-of-sample generalization.

3. **Missing-data handling (not implemented):** The specification calls for dropping any stock with > 10 missing factors per date, then zero-filling the rest. Currently, missing values are likely passed to the model, creating leakage if the model learns that NaN ≡ "no signal available."

**Model class and hyperparameters:**
- Base learner: LightGBM gradient boosted trees (assumed; not explicitly confirmed in research pack)
- Hyperparameters: not specified in research pack
- Feature importance method: SHAP kernel explainer
- Output: predicted alpha (long-short z-score)

**Rolling-window training:**
- Training window: 78 weeks (18 months)
- Validation window: 26 weeks (6 months)
- Retrain frequency: every 12 weeks (quarterly)
- This 3:1 train/val ratio is reasonable for a 3-year dataset; the paper uses a similar scheme on 19 years

**Portfolio construction:**
- Predict alpha for each stock each date
- Assign quintiles Q1 (weakest alpha) through Q5 (strongest alpha) across the universe
- **Current universe:** All S&P 500 constituents (≈ 500 names), not stratified by region or industry
- **Paper's approach:** Quintiles formed within region × industry peer groups, reducing cross-sector noise
- Go long Q5, short Q1; maintain equal weights within each quintile or dollar-neutral pairing
- Cost model: 1.5 bps round-trip per trade
- Execution lag: 1 day (respected in alpha-decay diagnostic; **not in main backtest**)

**Specification gaps:**
- **Neutralization not wired:** model sees raw features with sector bets
- **IREV1W placeholder:** currently = R1W; should be beta-adjusted residual
- **BETA6M missing:** can't compute market-neutral residuals
- **ADV filter disabled:** illiquid names included, slippage underestimated
- **Z-score capping (±3) missing:** outliers may have outsized influence
- **Weekday-effect diagnostic broken:** raises NotImplementedError
- **1-day lag only in alpha-decay diagnostic, not main backtest:** creates subtle timing mismatch
- **Earnings revisions absent:** cannot test UPDOWN1W (paper's #2 SHAP feature)

---

## Results

**No results are available.** The research pack confirms `pipeline_run_status: "no_run_yet"` and all output sections are marked `status: "not_yet_produced"`:
- `headline_metrics`
- `portfolio_summary`
- `predictions_summary`
- `shap_top_features`
- `alpha_decay`
- `weekday_effect`

Once the pipeline executes, the following metrics will be evaluated:

**Expected outputs and comparison framework:**

| Metric | Paper (US, 2006–2025) | This Run (S&P 500, 2023–2026) | Expected Gap Reason |
|--------|----------------------|-------------------------------|-------------------|
| Annualized return | 8.6% | TBD | Shorter sample, survivorship, missing earnings factors |
| Volatility | 8.5% | TBD | Lower in recent benign regime |
| Information ratio | 1.0 | TBD | Expect 0.6–0.8 (shorter window, model simplicity) |
| Max drawdown | −18% | TBD | 2023–2026 had lower realized vol |
| 2-way turnover | 212% | TBD | Expect similar or higher (no ADV scaling) |

**SHAP feature importance check:**
The paper reports R1W as the top ranked feature and UPDOWN1W as #2. This run will produce a SHAP ranking. We expect:
1. **R1W to rank #1** — it is the core alpha signal and the only strong price-action feature among the 12. Probability of confirmation: high (> 80%).
2. **UPDOWN1W to rank #2** — not possible; earnings revisions are absent from Phase 1. The #2 feature will be one of the volatility or momentum crosses. This **fails the validation check** by design.
3. **Volatility factors (IVOL, HVOL) to rank #3–5** — these capture micro-structure noise and may overfit.

**Alpha decay:**
The paper reports monotone decay from 14.9% (lag=0) to 6.8% (lag=4). The diagnostic will re-compute alpha assuming 1-day execution lag. We expect:
- Lag=0 (immediate fill): highest alpha, possibly 8–12% annualized (lower than paper due to shorter window and phase 1 simplicity)
- Lag=1,2,3,4: monotone decline to 4–6% as market-impact and information leakage erode the signal
- **Validation check: pass if monotone; fail if non-monotone or inverted**

**Weekday effect:**
The paper reports that Thursday signals capture 28% of data releases vs Monday's 8%, yielding a material Thursday outperformance. The weekday diagnostic in current code **raises NotImplementedError**, so this check cannot be evaluated. Placeholder: fail due to implementation gap.

**Quintile spread:**
Q5 (long) minus Q1 (short) should show monotonic cross-section alpha across all five quintiles. We expect a spread of 4–8% annualized within this sample. **Validation check:** pass if spreads are monotonic and statistically significant.

**Turnover:**
Paper reports 212% (ML) vs 317% (basic reversals). This run, without ADV-based position scaling, will likely show higher turnover (250–350% range). The comparison baseline (plain R1W) is not computed, so the "beat baseline" check cannot be passed.

---

## Limitations

1. **Survivorship bias:** The universe uses current Wikipedia constituents, excluding S&P 500 members that exited between 2023 and 2026 (mergers, bankruptcies, index drops). Empirically, this removes ~2–5% annualized drag from losses, inflating both realized return and information ratio. The paper reconstructs historical index membership; this run does not. **Impact: +50–100 bps IR bias.**

2. **Short sample window:** Three years vs the paper's 19 years. This run captures only one market regime (2023–2026: rising rates → falling rates → mega-cap concentration). Missing: 2008 GFC, 2011 US debt-ceiling crisis, 2015 China devaluation, 2020 COVID crash, 2022 energy and rate shock. Reversion alpha is cyclical (strong in mean-reversion regimes, weak in momentum/trend regimes). A 3-year window cannot discriminate cycle dependence. **Impact: high variance in estimated IR; 95% CI likely ±0.3–0.5 around point estimate.**

3. **Missing earnings revisions:** UPDOWN1W (paper's #2 SHAP feature) and SUE measures absent. These capture market underreaction to near-term earnings surprises, especially powerful when combined with price weakness. Omitting them reduces the model's feature count by 15%, forcing R1W to absorb all mean-reversion alpha. This mechanically inflates R1W's SHAP score. **Impact: R1W SHAP will rank higher than in the paper; model alpha lower by 20–30%.**

4. **Neutralization not wired:** Features fed raw, retaining industry and sector tilts. On a 3-year window dominated by mega-cap growth concentration, this allows the model to exploit sector rotation rather than true mean reversion. This inflates in-sample fit, likely harming out-of-sample generalization significantly. **Impact: estimated OOS IR may overstate true IR by 0.1–0.3 due to lucky regime fit.**

5. **Universe simplicity:** S&P 500 only; paper is global. US small-cap and mid-cap reversion is stronger than large-cap; global diversification reduces single-market regime bets. Also, S&P 500 is highly sector-concentrated in 2023–2026 (mega-cap tech), whereas the paper's global universe is more balanced. **Impact: R1W signal may be contaminated by cross-sector factors.**

6. **Ignored ADV filter:** The $3M ADV floor is not applied. Backtests will include highly illiquid small-caps and microcaps, whose 1.5 bps cost assumption is nonsensical. Actual slippage on microcap mean-reversion trades is 10–50 bps. **Impact: estimated turnover costs underestimated by 10–50 bps; OOS net return degraded 20–100 bps vs gross.**

7. **No ADV-based position scaling:** The paper applies position caps inversely proportional to ADV (e.g., $20M ADV → 1.0x position; $3M ADV → 0.2x). This reduces microcap tail risk. Current run maintains equal weight across all names. **Impact: max drawdown likely 20–50% deeper; Sharpe ratio degraded.**

8. **1-day lag inconsistency:** The alpha-decay diagnostic applies 1-day execution lag; the main backtest does not. This creates a timing mismatch where the headline metrics are over-optimistic. **Impact: reported net return overstated by 20–50 bps.**

9. **Simplified cost model:** 1.5 bps is a market-wide average. Real execution costs vary by ADV, time of day, and market conditions. For a high-turnover reversion strategy, costs are likely 2–3 bps average. **Impact: IR reduced by 0.05–0.15.**

10. **Z-score capping absent:** The paper applies ±3σ winsorizing. Without this, outliers can have disproportionate influence on model training, especially in low-liquidity periods. **Impact: out-of-sample feature importance estimates may be unstable.**

---

## Conclusion

**This backtest has not yet executed.** The research pack confirms that the pipeline is configured and ready to run, but all outputs remain unproduced. 

**What will this run tell us?**

If execution proceeds, the run will answer a narrower question than the paper: *Can mean-reversion alpha be extracted from price-action and volatility micro-structure alone, on a concentrated US large-cap universe over a 3-year benign-regime window, using a simplified feature set?*

The answer is likely **yes, but with material caveats:**

- R1W will almost certainly rank as the top SHAP feature (expected validation pass), confirming that price reversals remain primary.
- UPDOWN1W cannot be tested (Phase 1 limitation); the #2 feature will be a volatility or momentum cross, undershooting the paper's multi-factor hierarchy.
- Alpha decay will likely be monotone (expected pass), but absolute magnitudes will be lower (6–10% top-lag alpha vs paper's 15%).
- Turnover will be higher than the paper's 212%, absent ADV-based scaling.
- Information ratio will likely be 0.6–0.9, below the paper's 1.0, due to survivorship bias, missing fundamentals, short window, and neutralization gaps.
- Weekday diagnostics cannot be evaluated (NotImplementedError).

**Validation checklist tally:**
- Checks 1, 3–8: Cannot evaluate until run completes.
- Check 2 (UPDOWN1W #2): Will fail by design (Phase 1 omits earnings revisions).

**Next steps:**

1. **Execute the pipeline.** Run the backtest and collect headline metrics, SHAP rankings, alpha-decay curves, and portfolio summaries.
2. **Implement neutralization.** Wire the winsorize → z-score → industry-median pipeline into main.py. This is a critical gap.
3. **Add ADV filtering and position scaling.** Apply $3M ADV floor and scale positions inversely with ADV beyond $20M to match the paper.
4. **Fix IREV1W.** Compute true market-beta-adjusted residual returns, not raw R1W.
5. **Enable weekday diagnostic.** Unblock the NotImplementedError and quantify Thursday > Monday effect.
6. **Phase 2:** Add earnings-revision factors (UPDOWN1W, SUE1W, SUE3, SUE6) and re-run to test the #2 SHAP feature claim.
7. **Extend backtest window.** If data becomes available, re-test on 5–7 years to dampen survivorship-bias and regime-fit distortion.
8. **Establish baseline.** Build explicit R1W and earnings-filtered reversal baselines for direct head-to-head comparison.

The run will provide early validation of the core hypothesis (price reversals are predictive in an ML context), but structural constraints prevent full replication of the paper's findings. Interpretation should be cautious: a strong run on this config does not guarantee production viability on out-of-sample data, real market frictions, or different market regimes.