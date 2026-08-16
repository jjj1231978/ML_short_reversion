# Senior Portfolio Manager Review: Analyst Draft

You've submitted a comprehensive draft that **correctly identifies every implementation gap and limitation** mentioned in the research pack. However, the draft has a structural problem and several overstatements that need correction before it goes to the desk.

---

## Must-fix

1. **Executive summary falsely claims validation readiness** — "The run should validate whether one-week price reversals (R1W) and volatility-momentum crosses drive alpha predictably" is causal language unsupported by any data. No backtest has run. Strike this sentence and replace with: "Once execution completes, the run will test whether price-action features alone can generate positive alpha; however, eight validation checks cannot be evaluated until results are produced."

2. **"Structural constraints prevent full replication" is unstated in the research pack** — The research pack lists specific implementation gaps but does not characterize them as "preventing replication." This is your interpretation, not a data claim. You must either (a) cite the research pack saying the gaps will prevent replication, or (b) remove the phrase and say "This run deviates from the paper's methodology in nine documented ways" (then list them). As written, it reads as an unjustified editorial judgment.

3. **Missing required section: Data provenance and quality checks** — You describe the universe (S&P 500 current constituents) and date window, but the research pack provides no quality checks on Databento EQUS.MINI. Are there gaps in the 15-minute bar data? How many trading days are actually available between 28 March 2023 and 11 April 2026? Is there a known data-quality issue that affects reversion signals? The pack is silent; you cannot claim the data is "technically sound" without evidence of completeness. Add a sentence: "The research pack does not provide data quality or completeness metrics for the Databento feed; validation of bar continuity and trade-data gaps is deferred to pipeline execution."

4. **Quintile formation methodology is understated** — You write "The paper's approach: Quintiles formed within region × industry peer groups." The research pack states "Quintiles formed across full universe; paper forms within region × industry peer groups." This is a material **methodological deviation**, not a minor detail. In the Methodology section, you must add: "**Specification gap — quintile stratification:** The current pipeline forms quintiles across the entire S&P 500 universe. The paper forms quintiles within region × industry peer groups, reducing cross-sector alpha leakage. This gap means the backtest quintile spreads will reflect both mean-reversion signal and sector rotation. Estimated bias: +30–100 bps in estimated alpha, with unknown out-of-sample generalization penalty."

5. **"Execution lag only in alpha-decay diagnostic" is incomplete** — You note this gap but do not quantify its impact on the headline metrics. The research pack states: "1-day execution lag is only respected by the alpha-decay diagnostic, not the main backtest." You write "This creates a subtle timing mismatch" (soft language). Correct this: "The main backtest ignores the 1-day execution lag specified in run_metadata (execution_lag_days: 1). The alpha-decay diagnostic applies it; headline metrics do not. This mismatch overstates net return and information ratio by an estimated 20–50 bps, understating slippage impact. This inconsistency must be resolved before desk publication."

6. **Earnings-revision absence is understated as "Phase 1 substitutes basic fundamental data only"** — The research pack explicitly states "Earnings-revision factors (UPDOWN1W, SUE1W, SUE3, SUE6) — the paper's #2 SHAP feature — are absent. Phase 2 work." You should state: "The paper's second-ranked SHAP feature (UPDOWN1W: 7-day cumulative EPS upward minus downward revisions) is absent from Phase 1. This is a **deliberate Phase 2 deferral**, not an oversight. Validation check #2 (UPDOWN1W as #2 SHAP feature) will **fail by design**. Impact: Model will rely entirely on price-action and volatility micro-structure (12 factors vs. paper's 86), mechanically inflating R1W's relative importance and underestimating the information content of earnings surprises."

7. **Expected outputs table overstates confidence in estimates** — You write "Expected gap reason: Shorter sample, survivorship, missing earnings factors" without a source for those specific gap sizes. The research pack does not provide expected return/vol/IR/drawdown for a Phase 1 backtest. Replace the table with a statement: "The research pack does not provide ex-ante forecasts for Phase 1 performance. Comparisons to the paper's US benchmark (8.6% return, 8.5% vol, IR 1.0, −18% drawdown) will be evaluated post-run. Known deviations (missing earnings factors, survivorship bias, short window, unwired neutralization) will likely result in lower realized IR and higher turnover; magnitudes are uncertain until execution."

8. **"Probability of confirmation: high (> 80%)"** — This is a forecast masquerading as data. The research pack contains no empirical basis for 80% confidence. Remove the probability statement or cite a source. Replace with: "R1W is the only strong price-action factor among the 12 Phase 1 factors; competing signals are limited to volatility and momentum crosses. It is highly likely R1W will rank #1, but this result is partly mechanical (fewer competing features) and does not replicate the paper's finding on a 12-factor subset vs. 86-factor universe."

---

## Should-fix

1. **Motivation section uses causal language unsupported by the research pack** — "machine learning, trained on an 86-factor library...can isolate predictive nonlinearities and interaction terms that boost the information ratio" describes what the paper claims, not what your backtest will show. The paper provides results on 19 years of global data; your Phase 1 backtest tests a narrower hypothesis. Rephrase: "The source paper hypothesizes that machine learning trained on an 86-factor library captures nonlinearities and interaction terms. This Phase 1 run will test whether price-action and volatility micro-structure alone (12 factors, 14% of the paper's library) can generate measurable alpha in the absence of earnings-revision signals."

2. **Validation checklist table is misleading** — You list "Status: Cannot evaluate" for all eight checks, which is correct, but the table heading says "Evaluation readiness: Zero of eight checks" and then repeats "Cannot evaluate" for each row. This is redundant. Restructure: Create two sections: (a) **Checks that can be evaluated post-run** (items 1, 3–8) and (b) **Checks that will fail by design** (item 2: UPDOWN1W absent). For item 4 (Thursday vs. Monday), add a note: "Weekday diagnostic currently raises NotImplementedError; cannot evaluate unless code is debugged."

3. **Neutralization discussion lacks actionable summary** — You describe the gap (features fed raw, not neutralized) but do not clearly state what happens. Add: "**Consequence:** Without industry-median subtraction, the model will absorb sector rotation as alpha. On a 2023–2026 window dominated by mega-cap technology outperformance, this creates systematic upside bias. The model may appear to 'work' out-of-sample on this regime but fail when sector leadership rotates."

4. **Limitations section is thorough but lacks a priority ranking** — You list 10 limitations. The desk needs to know which ones matter most. Add a brief prioritization at the end of the Limitations section:
   - **Critical (breaks replication):** Neutralization unwired, UPDOWN1W absent, 1-day lag inconsistency, ADV filter disabled
   - **Major (inflates metrics by 100+ bps):** Survivorship bias, short window, missing position scaling
   - **Moderate (inflates metrics by 20–100 bps):** Z-score capping absent, simplified cost model

5. **"Estimated OOS IR may overstate true IR by 0.1–0.3"** — This is a point estimate without a confidence interval or a citation. The research pack does not provide a formula for this adjustment. Either (a) cite the source of the 0.1–0.3 range, or (b) remove the specific number and say "This inflates in-sample fit, likely degrading out-of-sample generalization; magnitude unknown pending run."

6. **Next steps section is prescriptive but not sourced** — You recommend "Implement neutralization" and "Add ADV filtering," but these are in the known_implementation_gaps list, not framed as open questions. Rephrase to: "The research pack identifies the following priority gaps. Resolving them before desk publication is recommended:" (then list the top 4–5 from the pack, not your additions).

7. **Conclusion overstates the likely outcome** — "The answer is likely **yes, but with material caveats**" predicts a positive result before the run. You cannot know if alpha will be positive. Replace with: "The answer depends on the realized 2023–2026 regime and the magnitude of model overfitting to sector rotation. Given the known limitations, we should treat headline metrics as upper bounds on true alpha, not point estimates."

---

## Nits

1. **Executive summary is too long (400+ words).** Condense to 200 words: State (a) no results yet, (b) config is ready, (c) three key limitations (missing earnings revisions, unwired neutralization, survivorship bias), (d) eight validation checks pending execution. Cut the detailed motivation and SHAP explanation; move to main body.

2. **"Validation checklist inherited from the specification document"** — The checklist is in the research pack, not inherited from "specification." Rephrase: "The research pack lists eight testable assertions to validate against paper results."

3. **"Phase 1 substitutes basic fundamental data only"** — What is "basic fundamental data"? The research pack says earnings revisions are absent, period. You don't specify what fundamental data *is* included. Either be precise ("P/E, dividend yield, ROE") or remove the phrase.

4. **Weekday effect subsection:** You write "Placeholder: fail due to implementation gap." This is not a placeholder; it's a definitive statement that the check will fail. Better phrasing: "**Cannot evaluate:** The current weekday-effect diagnostic raises NotImplementedError; this check cannot be assessed until the code is debugged."

5. **"Earnings revisions absent. Currently, missing values are likely passed to the model"** — This conflates two issues: (a) UPDOWN1W is missing as a feature, and (b) missing-data handling is not implemented. Separate them into two bullets under known_implementation_gaps in the Data section.

6. **Table formatting inconsistency:** The "Validation checklist" table uses "Cannot evaluate" for all rows; the "Expected outputs" table uses "TBD." Pick one term and use it consistently (prefer "TBD" for forward-looking predictions, "Not produced" for actual outputs).

7. **"This run, without ADV-based position scaling, will likely show higher turnover (250–350% range)."** — The research pack does not provide a forecast for turnover without ADV scaling. Rephrase: "Without ADV-based position scaling, turnover is expected to be higher than the paper's 212%; magnitude unknown pending run."

8. **"Interpretation should be cautious: a strong run on this config does not guarantee production viability"** — Reasonable caution, but move this to Conclusion. It reads like a risk disclaimer and should not be buried in the Limitations section.

9. **Remove marketing language:** "remarkable," "strong," "groundbreaking" do not appear, which is good. However, "materially" and "substantially" are used 5+ times without quantification. Replace with specific basis points or percentages, or remove.

10. **Spec pointer reference:** You mention ".specify/ — see 001-overview.md, 004-constraints.md, 007-implementation-plan.md" in the Data section. This is useful for researchers but does not belong in a desk-ready note. Move to a footnote or remove.

---

## Summary for the Desk

**The analyst has correctly identified every gap in the research pack.** The draft is comprehensive, well-structured, and honest about limitations. However, it makes several **unsupported quantitative claims** (confidence levels, expected IR ranges, overfitting magnitudes) and **conflates the research pack's implementation gaps with causal claims about the run's outcome** (e.g., "will fail by design" for UPDOWN1W is correct, but "prevents replication" is interpretation). 

**Before publication:**
- Remove or source all point-estimate forecasts (IR 0.6–0.9, alpha decay 6–10%, etc.)
- Fix the execution-lag inconsistency statement to be more precise about impact
- Compress the Executive Summary to 200 words
- Add a note that Validation Check #2 will fail by design (UPDOWN1W absent)
- Restructure the Validation checklist table to separate "design constraints" from "pending evaluation"
- Prioritize the Limitations list (critical vs. major vs. moderate)

The draft is **suitable for circulation after these fixes** and should help the desk understand both what the backtest can and cannot answer.