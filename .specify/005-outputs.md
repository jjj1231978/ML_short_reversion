# 005 — Outputs

What the pipeline produces, where it lives, and what to compare against.

## Artifact directory layout

```
data/                                                    # in-repo
└── processed/
    ├── portfolio_returns.parquet     ← weekly L/S returns + turnover
    ├── predictions.parquet           ← (date, ticker) → predicted score
    ├── shap_values.parquet           ← (date, ticker) × factor SHAPs (Phase-1.5)
    ├── feature_importance.parquet    ← mean |SHAP| per factor per retrain (Phase-1.5)
    ├── alpha_decay.parquet           ← lag → metrics table
    ├── weekday_effect.parquet        ← weekday → metrics table (Phase-1.5)
    ├── metrics_summary.json          ← top-level IR / return / DD / turnover
    └── reports/                      ← multi-agent research notes (see "Report generation" below)
        ├── latest → {run_id}/        ← symlink to most recent run
        └── {run_id}/                 ← one folder per report run, ISO timestamp
            ├── research_pack.json    ← structured input fed to all three agents
            ├── draft.md              ← briefing agent output
            ├── critique.md           ← review agent output
            ├── final.md              ← synthesis agent output (the "report")
            └── meta.json             ← model IDs, token counts, costs, timings

notebooks/
└── figures/                                             # PNGs from notebooks/diagnostics
    ├── sector_distribution.png
    ├── data_coverage.png
    ├── price_factor_distributions.png
    ├── price_factor_timeseries.png
    ├── factor_correlation.png
    ├── factor_correlation_timeseries.png
    ├── target_distribution.png
    ├── fundamental_distributions.png
    ├── shap_summary.png              ← Phase-1.5
    ├── shap_bar.png                  ← Phase-1.5
    ├── alpha_decay_curve.png         ← Phase-1.5
    ├── weekday_effect.png            ← Phase-2
    └── cumulative_ls_vs_baselines.png ← Phase-1.5
```

Raw data caches (Databento, SimFin) stay outside the repo, in `~/data_lake/`.
See [`002-requirements.md`](002-requirements.md) §2.

## Performance metrics produced per run

`compute_performance_metrics(portfolio)` returns:

| Key | Definition | Annualization |
|---|---|---|
| `annualized_return` | mean weekly L/S return × 52 | weekly → annual |
| `annualized_volatility` | std weekly L/S return × √52 | weekly → annual |
| `information_ratio` | ann. return / ann. vol | unitless |
| `max_drawdown` | minimum of cumulative_returns / running_max − 1 | unitless |
| `avg_weekly_turnover` | mean of (sym-diff / basket size), 2-way | unitless |
| `num_weeks` | observation count | count |

## Plots / tables expected

### Phase 1 (currently producible end-to-end, modulo the gaps in 007)

- Cumulative L/S equity curve (net of costs).
- Long-only, short-only, and L/S equity curves stacked.
- Per-quintile annualized return bar chart (Q1–Q5).
- Annual return / drawdown table by calendar year.

### Phase 1.5 (after wiring up SHAP and alpha-decay diagnostics)

- SHAP beeswarm summary plot for the most-recent retrain.
- Mean-|SHAP| feature importance bar chart.
- Alpha decay curve: IR vs execution lag (0–4 days).
- ML vs R1W-only baseline vs earnings-filtered baseline cumulative chart.
- 2-way weekly turnover comparison: ML vs baselines.

### Phase 2 (requires longer history + earnings revisions)

- Weekday-effect grid (5 cumulative curves, one per signal day).
- Crisis-period performance table (paper p.11 reproduction).
- Performance by market-return quintile bar chart (paper p.12 reproduction).
- 5-year rolling annualized return chart (paper p.11).
- 1-year rolling max drawdown chart (paper p.11).

## Reference numbers to compare against

US, after all costs and constraints, paper p.10:

| Metric | Paper |
|---|---|
| Annualized return | 8.6% |
| Annualized vol | 8.5% |
| Information ratio | 1.0 |
| Max drawdown | 18.0% |
| 2-way weekly turnover (Global) | 212% |
| Alpha at 0-day lag (Global) | 14.9% |
| Alpha at 1-day lag (Global) | 11.6% |
| Alpha at 4-day lag (Global) | 6.8% |

Use these as sanity bands for Phase 1, not exact targets — Phase 1 universe and
window differ.

## Streamlit app outputs

The Streamlit viewer (see [`006-streamlit-app.md`](006-streamlit-app.md)) is the
primary interactive consumption surface for these artifacts. It does not generate
new artifacts itself; it only reads from `data/processed/` and re-derives
parameter-explorer metrics on demand. The Streamlit "Report" page renders the
most-recent `final.md` produced by the agent workflow described below.

## Report generation (multi-agent)

Each pipeline run produces structured artifacts (parquets, JSON, plots). The
report layer turns those artifacts into a written research note via three
sequential LLM agents. The split — **draft → critique → synthesis** — exists to
catch the failure modes a single-shot writer is prone to: unfounded extrapolation,
arithmetic errors against the source data, missing caveats (survivorship, short
window), and tone drift from "research" toward "marketing."

### Agent flow

```
                         research_pack.json (structured facts)
                               │
                               ▼
                    ┌────────────────────┐
                    │  Briefing agent    │  → draft.md
                    │  (writes report)   │
                    └─────────┬──────────┘
                              │ draft.md + research_pack.json
                              ▼
                    ┌────────────────────┐
                    │  Critique agent    │  → critique.md
                    │  (challenges it)   │
                    └─────────┬──────────┘
                              │ draft.md + critique.md + research_pack.json
                              ▼
                    ┌────────────────────┐
                    │  Synthesis agent   │  → final.md (+ changelog footer)
                    │  (resolves)        │
                    └────────────────────┘
```

### Agent 1 — Briefing (writer)

**Role**: Quantitative research analyst writing a one-page note for an internal
audience.

**Input**: `research_pack.json` containing:
- Run metadata: `run_id`, date range, universe size, factor count.
- Headline metrics: `metrics_summary.json` contents.
- Top-N SHAP features (from `feature_importance.parquet`).
- Alpha-decay table.
- Weekday-effect table (if available).
- Per-quintile annualized return.
- Comparison reference values from the industry research paper (paper-Phase-1 deltas).
- Pointer to known limitations (forwarded from [`004-constraints.md`](004-constraints.md)).

**Output**: `draft.md`, target length **5-6 pages (~2500-3500 words)**, with the canonical section structure:

1. **Executive summary** (~150 words) — 3-5 bullets, headline IR + surprising finding + caveat + hypothesis read.
2. **Motivation** (~300 words) — why short-term equity reversion + why ML + what the industry research paper claims.
3. **Objective** (~250 words) — what the run is trying to test (the validation checklist from `004`).
4. **Data** (~350 words) — universe, window, source, filters, factor count vs. paper's 86.
5. **Methodology** (~500 words) — feature pipeline, neutralization recipe, model + windows, portfolio + costs; explicit on spec gaps.
6. **Results** (~700-900 words) — performance table with paper deltas, SHAP top-N with interpretation, alpha-decay, weekday effect, quintile spread.
7. **Limitations** (~400 words) — survivorship, sample length, missing factors, neutralization wiring, scope simplifications.
8. **Conclusion** (~250 words) — supports / refutes / inconclusive on the hypothesis, with next steps.

**System prompt sketch**:
> You are a buy-side quantitative researcher. You have just been handed the
> artifacts of a backtest run. Write a one-page research note. Be specific with
> numbers — never write "strong performance" without an IR. Compare against the
> reference paper values where applicable. Do not invent numbers; if a metric is
> missing from the research pack, say so explicitly.

### Agent 2 — Critique (reviewer)

**Role**: Senior PM giving the draft a hostile read before it goes out.

**Input**: `draft.md` + `research_pack.json` + a critique checklist:
- Every quantitative claim must trace to a value in the research pack — flag any that don't.
- Flag overclaims: small sample → confident generalization, single-period → secular trend.
- Flag missing context: any limitation in `004-constraints.md` that wasn't surfaced.
- Flag tone drift: "remarkable", "groundbreaking", etc.
- Flag missing comparisons: paper has reference numbers; were they used?
- Flag unsubstantiated causal language ("X drives Y" when only correlation is shown).

**Output**: `critique.md`, structured as:

```markdown
## Must-fix
1. <line/section reference> — <issue> — <suggested fix>
…

## Should-fix
1. …

## Nits
1. …
```

**System prompt sketch**:
> You are a skeptical senior PM reviewing a research note from a junior analyst.
> Your job is to find what they got wrong, what they overclaimed, and what they
> left out. Be specific — quote the offending sentence and explain the issue.
> Categorize each finding as Must-fix / Should-fix / Nit. Do not soften.

### Agent 3 — Synthesis (final author)

**Role**: Desk head producing the final version that incorporates the PM's
critique while preserving the analyst's substance.

**Input**: `draft.md` + `critique.md` + `research_pack.json`.

**Output**: `final.md` — the polished report — followed by a `## Changelog`
section listing each Must-fix and Should-fix critique with the resolution:
**incorporated** / **partially incorporated** / **rejected (with reason)**.

**System prompt sketch**:
> You are the desk head producing the final version of this research note.
> Take the analyst's draft and the PM's critique. For each Must-fix and
> Should-fix item: incorporate it, partially incorporate it, or reject it with
> a defensible reason. Output the final note, then a Changelog section listing
> each critique and your resolution. The final note must remain a one-pager —
> do not pad it.

### Iteration

Default: one round (draft → critique → synthesis).
Optional: `--rounds N` to loop critique→synthesis N times. Marginal value usually
drops off after round 2 — track it via the changelog density.

### Models — cheap by default

This pipeline is research/iteration, not a production deliverable. Defaults
should produce a usable report for **well under $0.05** per run. Quality
profiles exist for when the report actually matters.

API keys live in `.env` at the repo root (gitignored), loaded via `python-dotenv`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The orchestrator picks the provider per-agent based on the model ID prefix
(`claude-*` → Anthropic, `gpt-*` / `o*-` → OpenAI).

#### Profile presets

| Profile | Briefing | Critique | Synthesis | Approx cost / run |
|---|---|---|---|---|
| `cheap` (default) | `claude-haiku-4-5` | `claude-haiku-4-5` | `claude-haiku-4-5` | ~$0.10 |
| `cheap-openai` | `gpt-4o-mini` | `gpt-4o-mini` | `gpt-4o-mini` | ~$0.02 |
| `mixed` | `claude-haiku-4-5` | `gpt-4o-mini` | `claude-sonnet-4-6` | ~$0.15 |
| `quality` | `claude-opus-4-7` | `claude-sonnet-4-6` | `claude-opus-4-7` | ~$1.10 |
| `quality-openai` | `gpt-4o` | `gpt-4o-mini` | `gpt-4o` | ~$0.20 |

Costs are estimates against ~5K-token research pack and ~5K-token draft (5-6
page report). Actuals recorded in each run's `meta.json`. The default
`max_budget_usd` (in `configs/default.yaml`) is `0.20` — `quality` profile runs
must explicitly bump it.

#### Per-agent override

CLI flags trump profile selection. Use any model ID the provider serves:

```bash
python -m src.reporting.run_report                                     # cheap profile
python -m src.reporting.run_report --profile quality
python -m src.reporting.run_report --briefing-model claude-opus-4-7    # mix-in
python -m src.reporting.run_report --critique-model gpt-4o-mini --synth-model claude-sonnet-4-6
```

#### Config block

`configs/default.yaml` gets a new section (defaults shown):

```yaml
reporting:
  profile: cheap                    # cheap | cheap-openai | mixed | quality | quality-openai
  briefing_model: null              # explicit override; null = profile default
  critique_model: null
  synth_model: null
  rounds: 1                         # extra critique→synth iterations
  max_budget_usd: 0.10              # abort if estimated spend exceeds
  prompt_caching: true              # Anthropic-only optimization; ignored for OpenAI
```

#### Why these picks

- `claude-haiku-4-5` is the cheapest current Claude tier and writes coherent
  one-pagers given a tight system prompt — fine for iteration.
- `gpt-4o-mini` is cheaper still and a good cross-check (different model
  family → catches Claude-specific failure modes in critique).
- `claude-opus-4-7` for `quality` because writing+judgment is exactly what
  briefing and synthesis need. Sonnet handles critique well at ~5× lower cost.
- Prompt caching (Anthropic-only) reuses the `research_pack.json` block across
  the three calls when all three are Claude — meaningful saving with `quality`.

### Cost guard

Before each call, the orchestrator estimates input tokens against the chosen
model's price card. If the projected total for the run exceeds
`reporting.max_budget_usd`, the run aborts with a printed estimate. Final
`meta.json` records actual usage and dollar cost per agent.

### Implementation skeleton

```
src/reporting/
├── __init__.py
├── pack.py           ← assemble research_pack.json from data/processed/*
├── llm.py            ← provider-agnostic call(model, system, user) → text + usage
├── briefing.py       ← Agent 1
├── critique.py       ← Agent 2
├── synthesize.py     ← Agent 3
├── prompts/          ← system + user prompt templates
│   ├── briefing.md
│   ├── critique.md
│   └── synthesize.md
└── run_report.py     ← orchestrator: pack → brief → critique → synth → save
```

`llm.py` dispatches on model ID prefix:
- `claude-*` → `anthropic.Anthropic().messages.create(...)` with cache_control
  on the research pack block when `prompt_caching=true`.
- `gpt-*` / `o*-` → `openai.OpenAI().chat.completions.create(...)`.

Both SDKs read their key from environment by default after `dotenv.load_dotenv()`.

Add to `requirements.txt`:
```
anthropic>=0.40.0
openai>=1.50.0
python-dotenv>=1.0.0
```

### Run

```bash
python -m src.reporting.run_report                              # cheap profile, latest pipeline output
python -m src.reporting.run_report --profile quality
python -m src.reporting.run_report --rounds 2                   # extra critique→synth round
python -m src.reporting.run_report --run-id 2026-05-06T18-30 --dry-run
```

`--dry-run` assembles the research pack and prints the estimated cost without
calling any API.

### Failure modes to design against

- **Hallucinated metrics.** Mitigation: critique agent explicitly checks every
  number against the research pack; reject any that don't trace.
- **Critique nitpicks become noise.** Mitigation: the Must-fix / Should-fix / Nit
  tiering forces the synthesizer to prioritize.
- **Synthesizer rubber-stamps the draft.** Mitigation: changelog footer makes
  resolutions auditable; if every Must-fix is "rejected", investigate.
- **Cost runaway.** Mitigation: token budget guard above; cheap profile by default.
- **Single-vendor blind spot.** Mitigation: `mixed` profile uses both providers
  so the critique agent doesn't share the writer's training-data biases.
