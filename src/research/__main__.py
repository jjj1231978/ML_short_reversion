"""CLI for the per-name deep dive.

    # build the deterministic layer + fact sheets for the session to write up
    python -m src.research deepdive --signal-day THU

    # scope it down while iterating
    python -m src.research deepdive --signal-day THU --top-n 3 --regions US --sides long

    # cost/call estimate, no network
    python -m src.research deepdive --signal-day THU --dry-run

    # unattended: generate the briefs through the metered API instead
    python -m src.research deepdive --signal-day THU --llm-backend api

    # merge briefs written by the session back into the artifact
    python -m src.research merge-narratives --signal-day THU --file briefs.json

    # regenerate just the fact sheet pack from an existing artifact
    python -m src.research factsheets --signal-day THU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.research import deepdive as dd_mod
from src.research import narrative as nar

log = logging.getLogger(__name__)

# Per-name call counts, from the verified endpoint coverage table.
FMP_CALLS_US, FMP_CALLS_NON_US = 9, 6


def _resolve_target(args, dd: dict) -> str:
    """Explicit --target-date, else the newest forecast for that signal day."""
    if args.target_date:
        return args.target_date
    files = dd_mod.list_forecasts(args.signal_day)
    if not files:
        sys.exit(f"No forecasts for {args.signal_day.upper()}. "
                 f"Run: python -m src.predict --signal-day {args.signal_day.upper()}")
    return files[0].stem


def _write_factsheets(artifact: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nar.build_factsheet_pack(artifact))
    return path


def _apply_api_narratives(artifact: dict, dd: dict, max_workers: int) -> None:
    """Metered backend: one call per name, budget-guarded up front."""
    from src.reporting import llm

    model, recs = dd["llm_model"], artifact["tickers"]
    sample = nar.build_factsheet({**next(iter(recs.values())), "ticker": "SAMPLE"})
    est = llm.estimate_cost_usd(model, sample + nar.system_prompt(), 800) * len(recs)
    log.info(f"[api] {len(recs)} briefs via {model}; estimated upper bound ${est:.2f}")
    if est > dd["max_budget_usd"]:
        sys.exit(f"Estimated ${est:.2f} exceeds deepdive.max_budget_usd "
                 f"${dd['max_budget_usd']:.2f}. Raise the budget or lower --top-n.")

    def _one(item):
        ticker, rec = item
        return ticker, nar.generate_via_api({**rec, "ticker": ticker}, model)

    cost = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, (brief, usage, err) in pool.map(_one, list(recs.items())):
            if brief:
                recs[ticker]["narrative"] = brief
            if err:
                recs[ticker].setdefault("errors", []).append(err)
            cost += llm.actual_cost_usd(model, usage)

    n = sum(1 for r in recs.values() if r.get("narrative"))
    artifact["provenance"].update(
        llm_cost_usd=round(cost, 4), llm_model=model,
        n_narratives=n, narrative_coverage=f"{n}/{len(recs)}",
    )
    log.info(f"[api] {n}/{len(recs)} briefs; actual ${cost:.4f}")


def cmd_deepdive(args) -> None:
    dd = dd_mod.deepdive_config()
    target = _resolve_target(args, dd)
    top_n = args.top_n if args.top_n is not None else dd["top_n"]
    if top_n > dd["max_top_n"]:
        sys.exit(f"--top-n {top_n} exceeds deepdive.max_top_n ({dd['max_top_n']}).")

    regions = args.regions.split(",") if args.regions else dd["regions"]
    sides = args.sides.split(",") if args.sides else dd["sides"]
    tickers = args.tickers.split(",") if args.tickers else None

    if args.dry_run:
        import pandas as pd

        fc = pd.read_parquet(dd_mod.FORECASTS_ROOT / args.signal_day.upper() / f"{target}.parquet")
        sel = dd_mod.select_universe(fc, top_n, sides, regions, max_top_n=dd["max_top_n"])
        if tickers:
            sel = sel[sel["ticker"].isin(tickers)]
        n_us = int(sel["ticker"].map(lambda t: "." not in t).sum())
        fmp = n_us * FMP_CALLS_US + (len(sel) - n_us) * FMP_CALLS_NON_US
        print(f"target {target}: {len(sel)} names ({n_us} US, {len(sel) - n_us} non-US)")
        print(f"  FMP calls   ~{fmp} (cached calls are free; TTL 1-30d by endpoint)")
        print(f"  Tavily      ~{len(sel)}")
        print(f"  narratives  {len(sel)} via backend '{args.llm_backend or dd['llm_backend']}'")
        print(f"  est. FMP wall clock ~{fmp * 60 / dd['rate_limit_per_min'] / 60:.1f} min "
              f"at {dd['rate_limit_per_min']}/min")
        for region, grp in sel.groupby("region"):
            print(f"  {region}: {', '.join(grp['ticker'].head(6))}"
                  f"{' …' if len(grp) > 6 else ''}")
        return

    def _progress(i, n, ticker):
        log.info(f"[{i}/{n}] {ticker}")

    artifact = dd_mod.build_artifact(
        args.signal_day, target, top_n=top_n, sides=sides, regions=regions,
        tickers=tickers, force=args.force, progress=_progress,
    )

    backend = args.llm_backend or dd["llm_backend"]
    out = dd_mod.artifact_path(args.signal_day, artifact["target_date"], dd["output_dir"])

    # Merge onto an existing artifact so a single-ticker refresh, or a rerun
    # after the briefs are written, does not discard everything else.
    existing = dd_mod.load_artifact(out)
    if existing and existing.get("schema_version") == artifact["schema_version"]:
        for ticker, rec in existing.get("tickers", {}).items():
            if ticker not in artifact["tickers"]:
                artifact["tickers"][ticker] = rec
            elif rec.get("narrative") and not args.drop_narratives:
                artifact["tickers"][ticker]["narrative"] = rec["narrative"]

    if backend == "api":
        _apply_api_narratives(artifact, dd, args.max_workers or dd["max_workers"])
    elif backend == "none":
        log.info("[none] deterministic layer only — narratives left null")

    n = sum(1 for r in artifact["tickers"].values() if r.get("narrative"))
    artifact["provenance"]["n_narratives"] = n
    artifact["provenance"]["narrative_coverage"] = f"{n}/{len(artifact['tickers'])}"

    dd_mod.save_artifact(artifact, out)
    print(f"wrote {out}  ({len(artifact['tickers'])} names, {n} with a brief, "
          f"{artifact['provenance']['elapsed_seconds']}s)")

    if backend == "session":
        pack = _write_factsheets(
            artifact, dd_mod.factsheet_path(args.signal_day, artifact["target_date"], dd["output_dir"])
        )
        pending = [t for t, r in artifact["tickers"].items() if not r.get("narrative")]
        print(f"wrote {pack}")
        print(f"\nBackend 'session': {len(pending)} names still need a brief. Read the pack, "
              f"write one JSON object per ticker, then:\n"
              f"  python -m src.research merge-narratives --signal-day {artifact['signal_day']} "
              f"--target-date {artifact['target_date']} --file <briefs.json>")


def cmd_factsheets(args) -> None:
    dd = dd_mod.deepdive_config()
    target = _resolve_target(args, dd)
    path = dd_mod.artifact_path(args.signal_day, target, dd["output_dir"])
    artifact = dd_mod.load_artifact(path)
    if artifact is None:
        sys.exit(f"No artifact at {path}. Run `python -m src.research deepdive` first.")
    if args.pending_only:
        artifact = {**artifact, "tickers": {
            t: r for t, r in artifact["tickers"].items() if not r.get("narrative")
        }}
        if not artifact["tickers"]:
            print("Every covered name already has a brief.")
            return
    out = _write_factsheets(artifact, dd_mod.factsheet_path(args.signal_day, target, dd["output_dir"]))
    print(f"wrote {out}  ({len(artifact['tickers'])} names)")


def cmd_merge(args) -> None:
    dd = dd_mod.deepdive_config()
    target = _resolve_target(args, dd)
    path = dd_mod.artifact_path(args.signal_day, target, dd["output_dir"])
    artifact = dd_mod.load_artifact(path)
    if artifact is None:
        sys.exit(f"No artifact at {path}. Run `python -m src.research deepdive` first.")

    try:
        briefs = json.loads(Path(args.file).read_text())
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not read {args.file}: {exc}")
    if not isinstance(briefs, dict):
        sys.exit("Expected a JSON object mapping ticker -> brief.")

    merged, problems = nar.merge_narratives(artifact, briefs)
    artifact.setdefault("provenance", {})["llm_model"] = args.model or "claude-code-session"
    dd_mod.save_artifact(artifact, path)

    print(f"merged {merged} briefs into {path} "
          f"({artifact['provenance']['narrative_coverage']} covered)")
    for p in problems:
        print(f"  - {p}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.research", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--signal-day", default="THU", choices=["WED", "THU", "FRI"],
                        type=str.upper,
                        help="Thursday is the traded day and the only one the page reads")
        sp.add_argument("--target-date", help="YYYY-MM-DD; defaults to the newest forecast")

    b = sub.add_parser("deepdive", help="fetch fundamentals/news/technicals and write the artifact")
    _common(b)
    b.add_argument("--top-n", type=int, help="names per side per region (config cap applies)")
    b.add_argument("--sides", help="comma-separated: long,short")
    b.add_argument("--regions", help="comma-separated: US,UK,CA")
    b.add_argument("--tickers", help="comma-separated subset, for a single-name refresh")
    b.add_argument("--llm-backend", choices=["session", "api", "none"])
    b.add_argument("--max-workers", type=int)
    b.add_argument("--force", action="store_true", help="bypass the TTL cache")
    b.add_argument("--drop-narratives", action="store_true",
                   help="discard existing briefs instead of carrying them forward")
    b.add_argument("--dry-run", action="store_true", help="print the call estimate and exit")
    b.set_defaults(func=cmd_deepdive)

    f = sub.add_parser("factsheets", help="re-emit the fact sheet pack from an existing artifact")
    _common(f)
    f.add_argument("--pending-only", action="store_true", help="only names without a brief")
    f.set_defaults(func=cmd_factsheets)

    m = sub.add_parser("merge-narratives", help="validate and merge briefs into the artifact")
    _common(m)
    m.add_argument("--file", required=True, help="JSON object mapping ticker -> brief")
    m.add_argument("--model", help="recorded as provenance; defaults to claude-code-session")
    m.set_defaults(func=cmd_merge)
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":
    main()
