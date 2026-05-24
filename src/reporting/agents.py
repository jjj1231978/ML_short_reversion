"""The three agents: briefing → critique → synthesis.

Each agent loads its system prompt and dispatches via `llm.call`. The full
research pack is passed as `cached_prefix` so it's shared across all three
calls within a run via Anthropic prompt caching (cache key = prefix, which is
the pack — identical across agents). Per-agent `system` and `user` come after
the cached prefix and don't break the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting import llm
from src.reporting.pack import render_pack_markdown

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


def _setup_at_a_glance(pack: dict) -> str:
    """Render a compact Markdown summary of the run setup so the agent
    cannot miss it (otherwise the model pattern-matches on legacy framing)."""
    md = pack.get("run_metadata", {})
    regions = md.get("regions") or [md.get("config_universe_index", "?")]
    window = md.get("config_date_window", {})
    elig = md.get("eligibility_filter_components", [])
    headline = pack.get("headline_metrics", {})
    n_weeks = pack.get("portfolio_summary", {}).get("num_weeks") or headline.get("num_weeks")
    universe_method = md.get("universe_method", {})
    universe_lines = "\n".join(f"  - **{r}**: {v}" for r, v in universe_method.items())
    elig_lines = "\n".join(f"  - {c}" for c in elig) if elig else "  - (none — eligibility filter inactive)"
    return f"""## Run setup at a glance — READ THIS FIRST. Use these values verbatim. Do not substitute defaults from training.

- **Regions in this run**: {regions}
- **Date window**: {window.get('start')} → {window.get('end')}  (≈ {n_weeks} OOS weeks)
- **Factor count**: {md.get('factor_count')}
- **Universe method**:
{universe_lines}
- **Eligibility filter**:
{elig_lines}
- **Trading cost**: {md.get('trading_cost_bps')} bps/side, exec lag: {md.get('execution_lag_days')}d, weekly return cap: ±{md.get('weekly_return_cap')}
- **Train / val / retrain**: {md.get('config_train_weeks')} / {md.get('config_val_weeks')} / {md.get('config_retrain_freq')} weeks
- **Data vendor**: {md.get('data_vendor_summary')}
- **Price source**: `{md.get('price_source')}` · **Fundamental source**: `{md.get('fundamental_source')}` · **Universe source**: `{md.get('universe_source')}`
- **Earnings revisions source**: {md.get('earnings_revisions_source')}
- **Model**: `{md.get('model_class')}` of members {md.get('ensemble_members')} combined via `{md.get('ensemble_combine_method')}`. Headline metrics reflect the ensemble; per-member metrics are in `model_comparison`.

**If you write any of the following, you have ignored this section and the report is wrong:**
- "US-only" (when regions includes UK / CA)
- "3-year window" or "Phase 1" framing (unless `config_date_window` is actually <4 years)
- "current Wikipedia constituents" or "survivorship-biased universe" *applied to US* (US is point-in-time per universe_method above)
- Any factor count other than {md.get('factor_count')}
- "Phase 1" / "Phase 2" labels are spec-doc terminology — do not use them as if they describe THIS run; describe the run by its actual setup above.
- Any data vendor not listed in **Data vendor** above. Do NOT invent "Databento", "SimFin", "IBES", "Refinitiv", "Alpha Vantage", "Tiingo", "Wikipedia", or any other vendor — this run is FMP-exclusive.
- "XGBoost-only", "the XGBoost model", "we trained XGBoost". The headline run is a {md.get('model_class')} of {md.get('ensemble_members')}; XGBoost is one member of three. Always specify which member when discussing per-model SHAP rankings (top features in `shap_top_features_per_model` may differ across members — do not collapse them into "the model says X").

Headline metrics this run produced (from `metrics_summary.json`):

```json
{json.dumps(headline, indent=2, default=str)}
```
"""


def _pack_preface(pack: dict) -> str:
    """The cached prefix sent ahead of every agent's system prompt."""
    return (
        "# Research pack (source of truth for this run)\n\n"
        + _setup_at_a_glance(pack)
        + "\n\n## Full pack (JSON)\n\n"
        + "Treat the JSON below as authoritative; do not invent facts beyond it. "
        "Where the JSON conflicts with anything else (training data, paper text, prior reports), "
        "the JSON wins.\n\n"
        + render_pack_markdown(pack)
    )


def briefing(pack: dict, model: str, max_tokens: int = 10000) -> tuple[str, dict]:
    """Agent 1: generate the initial draft from the research pack."""
    system = _load_prompt("briefing")
    user = (
        "Write the research note per the system instructions. The research pack "
        "at the top of the system context is the source of truth."
    )
    return llm.call(model, system, user, max_tokens=max_tokens, cached_prefix=_pack_preface(pack))


def critique(
    pack: dict, draft: str, model: str, max_tokens: int = 4000
) -> tuple[str, dict]:
    """Agent 2: critique the draft against the research pack."""
    system = _load_prompt("critique")
    user = (
        "Here is the analyst's draft. Run the checks per the system instructions "
        "and return your findings.\n\n"
        "## Draft\n\n"
        f"{draft}"
    )
    return llm.call(model, system, user, max_tokens=max_tokens, cached_prefix=_pack_preface(pack))


def synthesize(
    pack: dict,
    draft: str,
    critique_text: str,
    model: str,
    max_tokens: int = 12000,
) -> tuple[str, dict]:
    """Agent 3: produce the final report incorporating critique."""
    system = _load_prompt("synthesize")
    user = (
        "Here is the analyst's draft and the PM's critique. Produce the final "
        "report per the system instructions.\n\n"
        "## Draft\n\n"
        f"{draft}\n\n"
        "## Critique\n\n"
        f"{critique_text}"
    )
    return llm.call(model, system, user, max_tokens=max_tokens, cached_prefix=_pack_preface(pack))
