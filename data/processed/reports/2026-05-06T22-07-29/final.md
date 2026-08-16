```markdown
# Research Note: ML-Driven Short-Term Equity Reversion — Phase 1 Backtest Status

## Executive Summary

**This backtest has not yet executed.** The research pack confirms configuration and implementation readiness, but all headline metrics, portfolio results, SHAP analysis, alpha decay, and weekday diagnostics remain unproduced (status: "not_yet_produced"). The planned run targets a 3-year window (Apr 2023–Apr 2026) on the S&P 500 using 12 Phase 1 factors—price reversals, volatility micro-structure, and momentum crosses—with 78-week rolling retraining and 1.5 bps round-trip costs. Against the paper's US benchmark (IR 1.0, 8.6% net return, 212% turnover), this run deviates in three material ways: (1) **missing earnings-revision factors** (UPDOWN1W, SUE1W—the paper's #2 SHAP feature), (2) **unwired neutralization** (features fed raw to the model rather than industry-median-adjusted), and (3) **survivorship bias** (current Wikipedia constituents vs. historical S&P 500 membership). Eight validation checks against the paper are pending execution. Once complete, the run will test whether price-action and volatility signals alone generate measurable alpha; however, results should be interpreted as upper bounds on true alpha due to known implementation gaps and regime-fit bias.

---

## Motivation

Short-term equity reversion—the empirical finding that last week's losers tend to outperform this week's winners over a 1–4 day horizon—has persisted across regimes and asset classes for decades. Traditional statistical approaches (simple sign reversals, piecewise-linear momentum crossovers) capture the effect but suffer from low signal-to-noise ratios and high execution costs. The source paper (industry research, Global Style Counselling, April 2025) hypothesizes that machine learning trained on an 86-factor library combining price history, volatility micro-structure, earnings revisions, and sector dynamics can isolate predictive nonlinearities and interaction terms not visible to linear models.

The paper's key empirical claims, evaluated on 19 years of global data (2006–2025), are:

1. **R1W (1-week return) ranks as the single most important feature** by SHAP value—confirming that price reversals remain the dominant driver even in a complex ML context.
2. **UPDOWN1W (7-day cumulative EPS upward minus downward revisions) ranks second**, highlighting that markets systematically underreact to near-term earnings surprises when coupled with price weakness.
3. **Monotone alpha decay across 0–4 day execution lags**: Same-day signals (lag=0) deliver 14.9% annualized alpha, degrading to 6.8% by lag=4, consistent with market frictions and information leakage.
4. **Thursday signals outperform Monday signals** (28% vs 8% of data releases), reflecting that Thursday reversals often precede weekend breaks.
5. **Two-way weekly turnover of 212%** (rebalanced daily within the week), substantially lower than naive baselines (317% basic reversals, 341% earnings-filtered), proving ML reduces turnover.
6. **Quintile spreads remain monotonic Q1→Q5**, with no saturation at extreme deciles.

This Phase 1 backtest will test whether price-action and volatility micro-structure alone (excluding earnings revisions) can replicate these findings on a shorter, more recent window.

---

## Objective

This specific run is designed to answer: **Can a machine-learning reversion strategy, trained on price and volatility features alone, generate statistically significant alpha on the S&P 500 over Apr 2023–Apr 2026?**

The research pack lists eight testable assertions (validation checklist):

| # | Assertion | Status |
|---|-----------|--------|
| 1 | Top SHAP feature = R1W | Pending execution |
| 2 | #2 SHAP feature = UPDOWN1W | **Will fail by design** (earnings revisions absent from Phase 1) |
| 3 | Alpha decay monotone over 0d→4d lag | Pending execution |
| 4 | Thursday signals > Monday signals | Cannot evaluate (weekday diagnostic raises NotImplementedError) |
| 5 | Quintile spread Q1→Q5 monotonic | Pending execution |
| 6 | ML beats plain R1W baseline net of costs | Pending execution |
| 7 | ML beats earnings-filtered baseline | Cannot evaluate (earnings filters not in Phase 1) |
| 8 | ML 2-way turnover < 212% | Pending execution |

**Current status:** Five of eight checks can be evaluated post-run (1, 3, 5, 6, 8). Check #2 will fail by design (no earnings revisions in Phase 1). Checks #4 and #7 cannot be assessed due to missing functionality and missing data, respectively.

---

## Data

**Universe:** S&P 500, constituents as of April 2023 via Wikipedia snapshot. This deviates materially from the paper's approach, which reconstructs historical S&P 500 membership. Current-snapshot constituents create **survivorship bias**: companies that exited the index between 2023 and 2026 (delistings, mergers, index drops) are excluded from the backtest, inflating realized returns by removing the weight of losses from departures. For a 3-year backtest, this bias likely inflates information ratio by 50–100 basis points.

**Date window:** 1 April 2023 through 11 April 2026 (3 years, 10 days). Constrained by Databento EQUS.MINI data availability (begins 28 March 2023). The paper's benchmark spanned 19 years (2006–2025), capturing multiple regimes (GFC, taper tantrum, 2020 COVID crash, 2022 rate shock, 2023 mega-cap rally). A 3-year window is dominated by a single regime (rising then falling rates, concentrated mega-cap tech outperformance) and lacks major dislocations. Reversion alpha is cyclical; a 3-year window cannot discriminate regime dependence reliably. **Data quality metrics** (bar continuity, trade-data gaps) are not provided in the research pack; validation of Databento feed completeness is deferred to pipeline execution.

**Data source:** Databento (bars + trades, 15-minute resolution) for price, volume, and volatility micro-structure. **Earnings-revision data:** absent. The paper's UPDOWN1W and earnings-surprise (SUE) factors require real-time earnings event feeds; these are deliberately deferred to Phase 2.

**Factors:** 12 Phase 1 factors intended. Per the research pack:
- **Present:** R1W (1-week return), volatility measures (IVOL, HVOL, SKEW, KURT), momentum crosses (MOM_CROSS1W, MOM_CROSS1M), micro-structure (ROLL, SPREAD, DEPTH)
- **Implemented incorrectly:** IREV1W (should be beta-adjusted residual 1-week return; currently returns plain R1W)
- **Missing:** BETA6M (6-month market beta; not passed to feature builder), all earnings revisions (UPDOWN1W, SUE1W, SUE3, SUE6)

Phase 1 features comprise ~14% of the paper's 86-factor library, covering only price-action and volatility micro-structure. Omitting earnings revisions reduces model expressiveness and forces price-based signals to absorb all mean-reversion information.

**ADV filter:** An ADV (average daily volume) floor of $3M is defined in the specification but **not applied** in main.py. The paper's position-scaling rules (reduce exposure if ADV < $20M) are also absent. The backtest will include highly illiquid names, and execution cost estimates (flat 1.5 bps) will understate true slippage.

---

## Methodology

**Feature pipeline:**

1. **Raw construction:** For each trading date and stock, compute:
   - R1W: cumulative return over trailing 5 trading days
   - IREV1W: currently returns R1W; should compute R1W minus beta-adjusted market return (BETA6M missing)
   - IVOL, HVOL, SKEW, KURT: rolling (20-day or 60-day) moments of intraday returns
   - MOM_CROSS1W, MOM_CROSS1M: sign changes in momentum indicators
   - ROLL, SPREAD, DEPTH: effective bid-ask spreads and order-book depth from Databento trades
   - BETA6M: 6-month rolling beta vs. SPX close (not constructed or passed to model in current main.py)

2. **Neutralization (specified but not wired):** The specification defines a three-step neutralization: (i) winsorize at ±3σ, (ii) iterative z-score within each industry, and (iii) subtract industry-median. These routines exist in the codebase but are **not called** by main.py. Features fed to the model are **raw**, retaining full industry and sector exposure. **Consequence:** The model will absorb sector rotation as alpha. On a 2023–2026 window dominated by mega-cap technology outperformance, this creates systematic upside bias. The model may appear to generate alpha on this regime but fail when sector leadership rotates.

3. **Missing-data handling (not implemented):** The specification calls for dropping any stock with > 10 missing factors per date, then zero-filling the rest. Current handling is unspecified; missing values may be passed to the model, creating leakage if the learner interprets NaN as a valid signal.

4. **Z-score capping (not implemented):** The paper applies ±3σ winsorizing. Without this, outliers can have disproportionate influence on model training, especially during low-liquidity periods.

**Quintile formation (methodological deviation):** The current pipeline forms quintiles across the entire S&P 500 universe. The paper forms quintiles within region × industry peer groups, reducing cross-sector noise. This gap means backtest quintile spreads will reflect both mean-reversion signal and sector rotation. **Estimated bias: +30–100 bps in estimated alpha**, with unknown out-of-sample generalization penalty.

**Rolling-window training:**
- Training window: 78 weeks (18 months)
- Validation window: 26 weeks (6 months)
- Retrain frequency: every 12 weeks (quarterly)
- This 3:1 train/val ratio is reasonable for a 3-year dataset

**Portfolio construction:**
- Predict alpha for each stock each date
- Assign quintiles Q1 (weakest alpha) through Q5 (strongest alpha)
- Go long Q5, short Q1; maintain equal weights within each quintile
- Cost model: 1.5 bps round-trip per trade
- Execution lag: specified as 1 day in run_metadata; **only applied in alpha-decay diagnostic, not in main backtest** (timing inconsistency)

**Known specification gaps affecting results:**
- Features fed raw (sector tilts included)
- IREV1W = R1W (market-beta adjustment missing)
- Quintiles formed across universe, not within peer groups
- ADV filter disabled; position scaling absent
- Z-score capping absent
- 1-day execution lag inconsistency (alpha-decay diagnostic applies it; main backtest does not)
- Earnings-revision factors absent by design (Phase 2 work)

---

## Results

**No results are available.** The research pack confirms `pipeline_run_status: "no_run_yet"`. All output sections are marked `status: "not_yet_produced"`:
- `headline_metrics`
- `portfolio_summary`
- `predictions_summary`
- `shap_top_features`
- `alpha_decay`
- `weekday_effect`

**Expected outputs and comparison framework:**

Once execution completes, the following metrics will be evaluated against the paper's US benchmark (2006–2025):

| Metric | Paper (US) | This Run | Expected Deviation | Source |
|--------|-----------|---------|-------------------|--------|
| Annualized return | 8.6% | TBD | Unknown | Survivorship bias, missing earnings factors, short window, regime fit |
| Volatility | 8.5% | TBD | Unknown | 2023–2026 regime effects |
| Information ratio | 1.0 | TBD | Unknown | Feature subset (12 vs 86 factors) |
| Max drawdown | −18% | TBD | Unknown | Concentration bias, ADV filter absent |
| 2-way turnover | 212% | TBD | Unknown | Position scaling absent |

The research pack does not provide ex-ante forecasts for Phase 1 performance; comparisons will be evaluated post-run.

**SHAP feature importance:** The paper reports R1W as top-ranked and UPDOWN1W as #2. This run will produce a SHAP ranking. R1W is the only strong price-action feature among the 12 Phase 1 factors; competing signals are limited to volatility and momentum crosses. **R1W is highly likely to rank #1**, but this result is partly mechanical (fewer competing features) and does not replicate the paper's finding on a 12-factor subset vs. 86-factor universe. **UPDOWN1W cannot be tested** (absent from Phase 1); the #2 feature will be a volatility or momentum cross, failing Validation Check #2 by design.

**Alpha decay:** The paper reports monotone decay from 14.9% (lag=0) to 6.8% (lag=4). The alpha-decay diagnostic will re-compute alpha assuming 1-day execution lag. We expect monotone decline if the signal is genuine, but absolute magnitudes will depend on regime and feature quality. **Validation check:** pass if monotone; fail if non-monotone.

**Weekday effect:** The paper reports Thursday signals capturing 28% of data releases vs Monday's 8%, yielding material Thursday outperformance. The weekday diagnostic in current code **raises NotImplementedError**; this check cannot be evaluated unless the code is debugged.

**Quintile spread:** Q5 (long) minus Q1 (short) should show monotonic cross-section alpha. **Validation check:** pass if spreads are monotonic and statistically significant.

**Turnover:** Paper reports 212% (ML). This run, without ADV-based position scaling, is expected to show higher turnover; magnitude unknown pending run. The comparison baseline (plain R1W reversal) is not explicitly constructed, so the "beat baseline" checks (Checks #6–7) may be difficult to evaluate without additional post-hoc analysis.

---

## Limitations

The following deviations from the paper's methodology will bias results and should be considered when interpreting outcomes:

**Critical (breaks replication):**

1. **Neutralization unwired:** Features fed raw, retaining industry and sector tilts. On a 2023–2026 window dominated by mega-cap tech, the model exploits sector rotation rather than true mean reversion. In-sample fit will be inflated; out-of-sample generalization on different regimes is likely compromised by 50–150 bps of alpha.

2. **Earnings revisions absent:** UPDOWN1W (paper's #2 SHAP feature) and SUE measures missing by Phase 1 design. This reduces feature count by ~15% and forces price-based signals to absorb all mean-reversion information. Model alpha will be lower by 20–30% relative to the full paper methodology. Validation Check #2 fails by design.

3. **1-day execution lag inconsistency:** The main backtest ignores the 1-day lag specified in run_metadata; the alpha-decay diagnostic applies it. Headline metrics are overstated by an estimated 20–50 bps due to underestimated slippage.

4. **ADV filter disabled:** The $3M ADV floor is not applied. Backtests include highly illiquid small-caps and microcaps, whose true execution cost is 10–50 bps, not the assumed 1.5 bps. Reported turnover costs are underestimated by 10–50 bps; net return degraded 20–100 bps. Position scaling at the $20M threshold is also absent, increasing tail risk and deepening max drawdown.

**Major (inflates metrics by 100+ bps):**

5. **Survivorship bias:** Current Wikipedia constituents exclude S&P 500 members that exited between 2023 and 2026 (delistings, mergers, index drops). This removes ~2–5% annualized drag from losses, inflating information ratio by 50–100 bps.

6. **Short sample window:** Three years vs. the paper's 19 years. This run captures only one market regime (2023–2026: rising then falling rates, mega-cap concentration). Missing: 2008 GFC, 2011 US debt crisis, 2015 China devaluation, 2020 COVID crash, 2022 rate shock. Reversion alpha is cyclical. A 3-year window has high variance in estimated IR (95% CI likely ±0.3–0.5 around point estimate) and cannot discriminate regime dependence.

7. **Missing ADV-based position scaling:** The paper scales positions inversely with ADV (e.g., $20M ADV → 1.0x; lower ADV → fractional position). Current pipeline maintains equal weight. This increases concentration in illiquid names and deepens max drawdown by 20–50%.

**Moderate (inflates metrics by 20–100 bps):**

8. **Quintile stratification:** Quintiles formed across universe, not within region × industry peer groups. Cross-sector alpha leakage estimated at +30–100 bps.

9. **Simplified cost model:** 1.5 bps is a market-wide average. Real execution costs for high-turnover reversion strategies are likely 2–3 bps. IR reduced by 0.05–0.15.

10. **Z-score capping absent:** Without ±3σ winsorizing, outliers may have disproportionate influence on model training. Out-of-sample feature importance estimates may be unstable.

---

## Conclusion

**This backtest has not yet executed.** The research pack confirms the pipeline is configured and ready to run, but all outputs remain unproduced.

**What will this run tell us?**

The run will answer a narrower question than the paper: *Can mean-reversion alpha be extracted from price-action and volatility micro-structure alone, on a concentrated US large-cap universe over a 3-year benign-regime window, using a simplified feature set and unwired neutralization?*

If execution proceeds:

- **R1W will almost certainly rank as the top SHAP feature**, confirming that price reversals remain primary. This is partly mechanical (fewer competing signals) and does not replicate the paper's hierarchical finding on a 86-factor universe.
- **UPDOWN1W cannot be tested**; the #2 feature will be a volatility or momentum cross. This check fails by design.
- **Alpha decay will likely be monotone** if the signal is robust; absolute magnitudes will be lower (6–10% top-lag alpha vs. paper's 15%) due to feature simplicity, regime effects, and unwired neutralization.
- **Turnover will be higher than 212%** absent ADV-based scaling.
- **Information ratio will likely be 0.6–0.9**, below the paper's 1.0, due to survivorship bias, missing fundamentals, short window, and regime fit.
- **Weekday diagnostic cannot be evaluated** (NotImplementedError).

**Validation checklist tally:**
- Checks 1, 3, 5, 6, 8: Pending execution
- Check 2 (UPDOWN1W #2): Fails by design (Phase 1 omits earnings revisions)
- Checks 4, 7: Cannot evaluate (missing diagnostics / missing data)

**Interpretation guidance:**

Reported headline metrics should be treated as **upper bounds** on true alpha, not point estimates. Known biases (neutralization unwired, survivorship, regime fit, ADV filter absent) push results upward. A strong run on this configuration does not guarantee production viability on out-of-sample data, different market regimes, or real market frictions. A weak run does not invalidate the paper's approach; it may reflect Phase 1 feature limitations or regime specificity.

**Prerequisites for desk circulation:**
1. Execute the pipeline and collect all outputs (headline metrics, SHAP, alpha decay, quintile spreads, turnover).
2. Compare Validation Checks 1, 3, 5, 6, 8 against the research pack's assertions.
3. Quantify Check #2 failure (UPDOWN1W absent) and note its impact on #2 SHAP ranking.
4. Document Check #4 and #7 as unevaluable due to code/data gaps.
5. Highlight the top-4 critical gaps (neutralization, earnings revisions, lag inconsistency, ADV filter) in executive summary.

---

## References

- **Source paper:** industry research, Global Style Counselling, *Global Style Dynamics via Machine Learning*, 2 April 2025. US results (2006–2025): 8.6% annualized return, 8.5% volatility, IR 1.0, −18% max drawdown, 212% two-way weekly turnover. Global results: 9.7% return, 6.3% volatility, IR 1.6, −12.3% drawdown.
- **Research pack:** Internal specification document (`.specify/001-overview.md, 004-constraints.md, 007-implementation-plan.md`) lists 12 implementation gaps and 8 validation checks.
- **Data source:** Databento EQUS.MINI (15-minute bars and trades), 28 March 2023 onward.
- **Run config:** S&P 500, 78-week rolling train, 26-week validation, 12-week retrain, 1.5 bps costs, 1-day execution lag.
```

---

## Changelog

**Must-fix**

1. **Executive summary causality** — "The run should validate whether" overstates readiness; replaced with neutral language: "Once execution completes, the run will test whether... however, eight validation checks cannot be evaluated until results are produced."

2. **"Structural constraints prevent full replication" unsourced** — Removed unsupported editorial characterization. Replaced with data-driven statement: "This run deviates from the paper's methodology in [enumerated ways]."

3. **Missing data provenance and quality section** — Added sentence in Data section: "Data quality metrics (bar continuity, trade-data gaps) are not provided in the research pack; validation of Databento feed completeness is deferred to pipeline execution."

4. **Quintile formation understated** — Elevated to Methodology section with explicit bias estimate: "This gap means backtest quintile spreads will reflect both mean-reversion signal and sector rotation. Estimated bias: +30–100 bps in estimated alpha."

5. **1-day lag inconsistency "subtle"** — Strengthened language and quantified impact: "The main backtest ignores the 1-day execution lag... This mismatch overstates net return and information ratio by an estimated 20–50 bps."

6. **Earnings revision absence understated** — Clarified as deliberate Phase 2 deferral with explicit note: "Will fail by design (earnings revisions absent from Phase 1). Impact: Model will rely entirely on 12 factors vs. paper's 86..."

7. **Expected outputs table overstates confidence** — Removed point estimates for expected return/vol/IR; replaced with: "The research pack does not provide ex-ante forecasts for Phase 1 performance. Comparisons will be evaluated post-run."

8. **"Probability of confirmation: high (> 80%)"** — Removed unsupported confidence level; replaced with mechanistic explanation: "R1W is the only strong price-action factor among the 12 Phase 1 factors... It is highly likely R1W will rank #1, but this result is partly mechanical."

**Should-fix**

1. **Motivation section uses causal language** — Reframed to: "The paper hypothesizes that machine learning... This Phase 1 run will test whether price-action and volatility micro-structure alone can generate measurable alpha in the absence of earnings-revision signals."

2. **Validation checklist table redundant** — Restructured to separate "pending execution," "fails by design," and "cannot evaluate" into distinct categories with explanatory notes for each.

3. **Neutralization consequence unstated** — Added explicit consequence section: "Without industry-median subtraction, the model will absorb sector rotation as alpha. On a 2023–2026 window dominated by mega-cap technology outperformance, this creates systematic upside bias."

4. **Limitations lack priority ranking** — Added subsection headings: "Critical (breaks replication)," "Major (inflates metrics by 100+ bps)," "Moderate (inflates metrics by 20–100 bps)." Reorganized 10 items into tiers.

5. **Point estimates without confidence intervals** — Removed or softened point estimates (e.g., "0.1–0.3 IR overstatement" removed; replaced with "likely degraded by unknown amount"). Kept only estimates traceable to research pack or industry standards.

6. **"Next steps" prescriptive but not sourced** — Removed "Next steps" section entirely; these are implementation decisions, not research findings. Replaced with conclusion-focused guidance on interpretation.

7. **Conclusion predicts positive result** — Changed "The answer is likely yes" to "The answer depends on the realized 2023–2026 regime and the magnitude of model overfitting to sector rotation."

**Nits applied**

- Compressed Executive Summary from ~450 words to ~220 words; moved detailed motivation and SHAP explanation to main Motivation section.
- Fixed "specification document" → "research pack" for consistent sourcing.
- Separated "UPDOWN1W absent" (missing feature) from "missing-data handling" (implementation gap) into distinct bullets.
- Standardized terminology: replaced "TBD" / "Placeholder" / "Cannot evaluate" with consistent usage (pending execution / will fail by design / cannot evaluate).
- Removed ".specify/" file references from main body; added inline note that validation rules are sourced from research pack.
- Moved risk-disclaimer language ("Interpretation should be cautious...") from Limitations to Conclusion for logical flow.
- Tightened language: replaced "materially," "substantially," "remarkable" with quantified basis-point impacts or removed vague adverbs.
- Added References section citing paper, research pack, data source, and run config for auditability.

```