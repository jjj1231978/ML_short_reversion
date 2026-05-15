# 006 — Streamlit App

Interactive viewer with on-the-fly parameter exploration. Read-only with respect to
the heavy ML steps (no model retraining from the UI), but it lets the user re-run
portfolio construction with different cost / quintile / lag knobs.

## Scope

| In | Out |
|---|---|
| Render performance, SHAP, alpha decay, weekday effect from cached parquets | Triggering data fetch |
| Recompute `build_long_short_portfolio` with user-supplied cost bps / quintile / lag | Triggering model retrain |
| Cross-filter SHAP plots by date / ticker | Modifying source data |
| Compare two parameter scenarios side by side | Long-running operations (>~30s) in the UI |

## Layout

A multi-page Streamlit app (`st.navigation` with `st.Page` entries — Streamlit ≥ 1.36).

```
app/
├── streamlit_app.py            ← main entrypoint, navigation
├── pages/
│   ├── 01_overview.py          ← headline metrics + cumulative L/S chart
│   ├── 02_predictions.py       ← per-week prediction histogram, top/bottom names
│   ├── 03_shap.py              ← SHAP beeswarm + feature importance + drilldown
│   ├── 04_alpha_decay.py       ← lag curve + table
│   ├── 05_weekday_effect.py    ← per-weekday cumulative + table
│   ├── 06_parameter_explorer.py ← sliders for cost/quintile/lag, recomputes metrics
│   └── 07_report.py            ← rendered final.md from latest report run
├── lib/
│   ├── data_loader.py          ← cached parquet loaders (st.cache_data)
│   ├── portfolio_recompute.py  ← thin wrapper over build_long_short_portfolio
│   └── plots.py                ← Plotly chart helpers
└── README.md                   ← run instructions, screenshot
```

## Page details

### 01 — Overview

- Top row: 4 metric cards (Ann. Return, IR, Max DD, Avg Turnover).
- Cumulative net L/S equity curve (Plotly line).
- Optional overlay toggles for Long-only, Short-only, ML-vs-R1W baseline.
- Date range slider in the sidebar.

### 02 — Predictions

- Date selector → cross-sectional histogram of predicted scores for that week.
- Top-10 long picks and bottom-10 short picks, with their predicted score and
  realized 1-week return.
- Cumulative IC plot (Spearman correlation between predictions and realized returns
  per week, then cumulative).

### 03 — SHAP

- Beeswarm summary plot (matplotlib via `shap.summary_plot`, embedded as image).
- Mean-|SHAP| bar chart (Plotly).
- "Drilldown" tab: pick a (date, ticker) → waterfall of feature contributions.
- Reads from `data/processed/shap_values.parquet`.

### 04 — Alpha decay

- Line chart: IR vs execution lag (0–4 days).
- Table view: full metrics per lag.
- Reads from `data/processed/alpha_decay.parquet`.

### 05 — Weekday effect

- 5 cumulative L/S curves (Mon–Fri signal days).
- Bar chart: annualized return per signal day.
- Reads from `data/processed/weekday_effect.parquet`. Hidden / stub until Phase 2
  produces it.

### 07 — Report

Renders the most recent `final.md` produced by the multi-agent reporting workflow
(see [`005-outputs.md`](005-outputs.md) "Report generation"). Sidebar selector
to pick a specific historical run from `data/processed/reports/`. Tabs:

- **Final** — the polished report (`final.md`) rendered as markdown.
- **Draft** — the briefing agent's first pass (`draft.md`).
- **Critique** — the review agent's findings (`critique.md`).
- **Meta** — model IDs used, token counts, dollar cost, timing (from `meta.json`).
- **Diff** — text diff of draft vs final, so you can see what the synthesizer
  changed in response to the critique.

Read-only; does not trigger new report runs (those are CLI:
`python -m src.reporting.run_report`). A "Run new report" button could be added
later but would block the UI for ~30s and risks API spend from the browser.

### 06 — Parameter explorer (interactive)

Inputs (sidebar):

| Knob | Range | Effect |
|---|---|---|
| Trading cost (bps/side) | 0.0 – 5.0 (slider) | Re-applies cost penalty to existing turnover series |
| Long quintile | 4, 5 (radio) | Re-runs `assign_quintiles` and `build_long_short_portfolio` |
| Short quintile | 1, 2 (radio) | Same |
| Execution lag (days) | 0 – 4 (slider) | Shifts the weekly return series before computing P&L |
| Date window | min – max (range slider) | Subsets predictions and returns |

Outputs: live-updating metrics card row + cumulative chart. Cached predictions and
weekly returns are loaded once via `st.cache_data`; recompute fits in <1s for
typical Phase-1 universe.

Save / Compare button: snapshot current metrics into a comparison table at the
bottom; up to 4 scenarios visible side-by-side.

## Tech choices

- **Streamlit** ≥ 1.36 (multi-page via `st.navigation`).
- **Plotly** for interactive line / bar charts (already in `requirements.txt`).
- **Matplotlib + SHAP** for the beeswarm (SHAP's plot returns a matplotlib figure).
- **`st.cache_data`** keyed by file mtime so re-running the pipeline auto-invalidates.
- **No backend service**: read parquet, compute in-process. If recompute ever
  exceeds ~3s, switch to `st.cache_data` keyed by parameter tuple.

## Run

```bash
streamlit run app/streamlit_app.py
```

Add `streamlit>=1.36` to `requirements.txt` when implementing.

## Out of scope

- Authentication / multi-user state.
- Live data fetch buttons (would block the UI for minutes).
- Model retraining buttons (multi-minute compute; hide behind CLI instead).
- Writing back to disk from the UI.
