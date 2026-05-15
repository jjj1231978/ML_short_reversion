# ML Short Reversion — Spec

This directory documents the research objective, requirements, procedures, constraints,
expected outputs, the planned Streamlit visualization, and the implementation
roadmap for the project.

Source paper: [`../ML_Mean_Reversion.pdf`](../ML_Mean_Reversion.pdf) — Société Générale,
"Can ML help us design a better stock mean reversion strategy?", 2 April 2025.

Read in this order:

1. [`001-overview.md`](001-overview.md) — research objective, hypothesis, phased scope
2. [`002-requirements.md`](002-requirements.md) — universe, data, factor, model, backtest specs
3. [`003-procedures.md`](003-procedures.md) — pipeline phases, formulas, ordering
4. [`004-constraints.md`](004-constraints.md) — biases, look-ahead, validation rules
5. [`005-outputs.md`](005-outputs.md) — artifacts, metrics, comparison references
6. [`006-streamlit-app.md`](006-streamlit-app.md) — viewer + parameter explorer design
7. [`007-implementation-plan.md`](007-implementation-plan.md) — done / broken / next, with ordering

Status: **Phase 1 shipped (2026-05-15).** End-to-end pipeline runs on 955 weeks
(2008-2026) of US/UK/CA data, 24 factors, ensemble model, Streamlit viewer, and
multi-agent report layer. Paper as of 2025-04-02. See [`007`](007-implementation-plan.md)
for the live status table and Phase 2 priorities (earnings revisions, region ×
industry peer grouping, factor expansion).

Phase 2 expands toward the paper's full global / 86-factor scope. The Phase 1
results (in 007) include a notable deviation from the paper's central claim —
R1W ranks #4 by SHAP, not #1 — which makes earnings-revision sourcing the
load-bearing Phase 2 experiment.
