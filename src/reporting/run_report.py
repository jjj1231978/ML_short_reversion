"""Orchestrate a multi-agent report run: pack → brief → critique → synth → save.

Usage:
    python -m src.reporting.run_report                       # cheap profile
    python -m src.reporting.run_report --profile quality
    python -m src.reporting.run_report --briefing-model claude-opus-4-7
    python -m src.reporting.run_report --rounds 2
    python -m src.reporting.run_report --dry-run             # estimate cost only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.reporting import agents, llm
from src.reporting.pack import build_research_pack, render_pack_markdown

log = logging.getLogger(__name__)


# Profile presets. Override any per-agent value via CLI.
PROFILES = {
    "cheap": {
        "briefing": "claude-haiku-4-5",
        "critique": "claude-haiku-4-5",
        "synth": "claude-haiku-4-5",
    },
    "cheap-openai": {
        "briefing": "gpt-4o-mini",
        "critique": "gpt-4o-mini",
        "synth": "gpt-4o-mini",
    },
    "mixed": {
        "briefing": "claude-haiku-4-5",
        "critique": "gpt-4o-mini",
        "synth": "claude-sonnet-4-6",
    },
    "quality": {
        "briefing": "claude-opus-4-7",
        "critique": "claude-sonnet-4-6",
        "synth": "claude-opus-4-7",
    },
    "quality-openai": {
        "briefing": "gpt-4o",
        "critique": "gpt-4o-mini",
        "synth": "gpt-4o",
    },
}


def resolve_models(cfg: dict, args: argparse.Namespace) -> dict[str, str]:
    """Profile + per-agent overrides → final {briefing, critique, synth} model ids."""
    rcfg = cfg.get("reporting", {})
    profile = args.profile or rcfg.get("profile", "cheap")
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown profile {profile!r}. Choose from {sorted(PROFILES)}."
        )
    models = dict(PROFILES[profile])

    # Config-level overrides (yaml)
    for k_cfg, k_local in [
        ("briefing_model", "briefing"),
        ("critique_model", "critique"),
        ("synth_model", "synth"),
    ]:
        v = rcfg.get(k_cfg)
        if v:
            models[k_local] = v

    # CLI overrides win
    if args.briefing_model:
        models["briefing"] = args.briefing_model
    if args.critique_model:
        models["critique"] = args.critique_model
    if args.synth_model:
        models["synth"] = args.synth_model

    return models


def estimate_total_cost(pack_md: str, models: dict[str, str], rounds: int) -> dict:
    """Conservative upper-bound cost estimate, assuming no cache hits.

    Real cost will be lower for Anthropic models because the pack is sent as a
    cached prefix and reused across critique + synth calls.
    """
    OUT_TOKENS = {"briefing": 5000, "critique": 3000, "synth": 6000}
    USER_TOKENS = {"briefing": 100, "critique": 5000, "synth": 8000}
    AGENT_PROMPT_TOKENS = 2000  # rough size of each per-agent system prompt

    breakdown = {}
    total = 0.0
    pack_tokens = llm.estimate_tokens(pack_md)

    for role in ("briefing", "critique", "synth"):
        # Total input = cached_prefix (pack) + per-agent system + user message
        in_tokens = pack_tokens + AGENT_PROMPT_TOKENS + USER_TOKENS[role]
        in_text_proxy = "X" * (in_tokens * 4)  # for estimate_cost_usd
        c = llm.estimate_cost_usd(models[role], in_text_proxy, OUT_TOKENS[role])
        breakdown[role] = {
            "model": models[role],
            "est_input_tokens": in_tokens,
            "est_output_tokens": OUT_TOKENS[role],
            "est_cost_usd": round(c, 4),
        }
        total += c * (rounds if role != "briefing" else 1)

    breakdown["total_estimated_usd"] = round(total, 4)
    breakdown["rounds"] = rounds
    breakdown["note"] = (
        "Upper bound: ignores Anthropic prompt-caching savings. Actual cost on "
        "claude-* models with the pack cached is typically 30-40% lower."
    )
    return breakdown


def run(args: argparse.Namespace) -> int:
    cfg = load_config()
    rcfg = cfg.get("reporting", {})
    rounds = args.rounds or rcfg.get("rounds", 1)
    max_budget = args.max_budget_usd or rcfg.get("max_budget_usd", 0.20)

    models = resolve_models(cfg, args)
    log.info(f"Models: {models}")
    log.info(f"Rounds: {rounds}, max_budget_usd: ${max_budget:.2f}")

    # Build pack
    pack = build_research_pack()
    pack_md = render_pack_markdown(pack)
    log.info(f"Research pack: {len(pack_md):,} chars (~{llm.estimate_tokens(pack_md):,} tokens)")
    log.info(f"Pipeline run status: {pack['pipeline_run_status']}")

    # Cost guard
    estimate = estimate_total_cost(pack_md, models, rounds)
    log.info(f"Estimated cost: ${estimate['total_estimated_usd']:.4f}")
    log.info(f"  Breakdown: briefing=${estimate['briefing']['est_cost_usd']}, "
             f"critique=${estimate['critique']['est_cost_usd']} x{rounds}, "
             f"synth=${estimate['synth']['est_cost_usd']} x{rounds}")

    if args.dry_run:
        print(json.dumps(estimate, indent=2))
        return 0

    if estimate["total_estimated_usd"] > max_budget:
        log.error(
            f"Estimated cost ${estimate['total_estimated_usd']:.4f} exceeds "
            f"budget ${max_budget:.2f}. Bump reporting.max_budget_usd or pick "
            f"a cheaper profile. Aborting."
        )
        return 2

    # Output dir
    run_id = args.run_id or dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_root = PROJECT_ROOT / rcfg.get("output_dir", "data/processed/reports")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing to {out_dir}")

    # Save the research pack first so we have something even if an agent fails
    (out_dir / "research_pack.json").write_text(json.dumps(pack, indent=2, default=str))

    usages: dict = {"briefing": None, "critique": [], "synth": []}

    # Briefing
    log.info(f"[briefing] Calling {models['briefing']}...")
    draft, br_usage = agents.briefing(pack, models["briefing"])
    (out_dir / "draft.md").write_text(draft)
    usages["briefing"] = br_usage
    log.info(f"[briefing] {br_usage['input_tokens']} in, {br_usage['output_tokens']} out, "
             f"{br_usage['elapsed_seconds']}s, "
             f"actual ${llm.actual_cost_usd(models['briefing'], br_usage):.4f}")

    current = draft
    last_critique = ""
    for r in range(rounds):
        log.info(f"[round {r+1}/{rounds}] critique → synth")

        # Critique
        log.info(f"[critique] Calling {models['critique']}...")
        crit_text, cr_usage = agents.critique(pack, current, models["critique"])
        crit_path = out_dir / (f"critique_round{r+1}.md" if rounds > 1 else "critique.md")
        crit_path.write_text(crit_text)
        usages["critique"].append(cr_usage)
        log.info(f"[critique] {cr_usage['input_tokens']} in, {cr_usage['output_tokens']} out, "
                 f"{cr_usage['elapsed_seconds']}s, "
                 f"actual ${llm.actual_cost_usd(models['critique'], cr_usage):.4f}")
        last_critique = crit_text

        # Synthesis
        log.info(f"[synth] Calling {models['synth']}...")
        final_text, sy_usage = agents.synthesize(pack, current, crit_text, models["synth"])
        synth_path = out_dir / (f"final_round{r+1}.md" if rounds > 1 else "final.md")
        synth_path.write_text(final_text)
        usages["synth"].append(sy_usage)
        log.info(f"[synth] {sy_usage['input_tokens']} in, {sy_usage['output_tokens']} out, "
                 f"{sy_usage['elapsed_seconds']}s, "
                 f"actual ${llm.actual_cost_usd(models['synth'], sy_usage):.4f}")
        current = final_text

    # Always make the canonical paths point at the last round's output
    if rounds > 1:
        (out_dir / "final.md").write_text(current)
        (out_dir / "critique.md").write_text(last_critique)

    # Compute total actual cost (accounts for cache write/read pricing)
    actual_total = llm.actual_cost_usd(models["briefing"], br_usage)
    for u in usages["critique"]:
        actual_total += llm.actual_cost_usd(u["model"], u)
    for u in usages["synth"]:
        actual_total += llm.actual_cost_usd(u["model"], u)

    meta = {
        "run_id": run_id,
        "models": models,
        "rounds": rounds,
        "estimated_cost_usd": estimate["total_estimated_usd"],
        "actual_cost_usd": round(actual_total, 4),
        "budget_usd": max_budget,
        "pipeline_run_status": pack["pipeline_run_status"],
        "usages": usages,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Update `latest` symlink
    latest = out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_id)

    log.info(f"Done. Total actual cost ${actual_total:.4f} (estimate was ${estimate['total_estimated_usd']:.4f}).")
    log.info(f"Final report: {out_dir / 'final.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-agent report.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--briefing-model", default=None)
    parser.add_argument("--critique-model", default=None)
    parser.add_argument("--synth-model", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--run-id", default=None, help="Override the auto timestamp run id.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost and exit without calling any API.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
