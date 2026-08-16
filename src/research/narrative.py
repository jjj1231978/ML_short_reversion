"""The narrative layer: fact sheet in, validated JSON brief out.

Three backends, chosen with `--llm-backend`:

  session  (default)  Emit fact sheets to a markdown pack and stop. The Claude
                      Code session writes the briefs and merges them back with
                      `merge-narratives`. Costs nothing beyond the session that
                      is already running, and uses a stronger model than the
                      metered path would be worth.
  api                 Call the Anthropic/OpenAI API once per name through
                      `src.reporting.llm`. Unattended — this is the backend a
                      cron job needs, since `session` requires a human-driven
                      session in the loop.
  none                Deterministic layer only; `narrative` stays null and the
                      page falls back to the Tavily summary plus the computed
                      technical verdict.

Both generating backends share `build_factsheet`, the same system prompt, and
the same validator, so a brief produced in-session is interchangeable with one
produced through the API.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "deepdive.md"

REQUIRED_STR_KEYS = ("thesis", "theme", "fundamental_read", "technical_read")
LIST_KEY_LIMITS = {"headline_risks": 4, "catalysts": 3, "sources_used": 12}
ENUMS = {
    "model_agreement": {"agrees", "partly", "disagrees"},
    "confidence": {"low", "medium", "high"},
}
MAX_THESIS_WORDS = 25


def system_prompt() -> str:
    return PROMPT_PATH.read_text()


# --- fact sheet ---------------------------------------------------------------


def _fmt(v, spec: str = ",.2f", dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return dash
    if isinstance(v, (int, float)):
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def _kv_block(title: str, pairs: list[tuple[str, str]]) -> str:
    rows = [f"- {k}: {v}" for k, v in pairs if v not in ("—", None)]
    return f"### {title}\n" + ("\n".join(rows) if rows else "- not available") + "\n"


def build_factsheet(rec: dict) -> str:
    """One name's supplied facts, as compact markdown.

    This is the *entire* evidence base for the brief — whatever is omitted here
    the model is required to treat as unknown, so err toward including a field
    with an em-dash over dropping it silently.
    """
    meta = rec.get("meta") or {}
    fun = rec.get("fundamentals") or {}
    tech = rec.get("technicals") or {}
    news = rec.get("news") or {}
    cur = meta.get("currency") or ""

    le = fun.get("last_earnings") or {}
    ne = fun.get("next_earnings") or {}
    val = fun.get("valuation") or {}
    est = fun.get("estimates_next_fy") or {}
    cons = fun.get("consensus") or {}

    out = [
        f"# {meta.get('company_name') or rec.get('ticker')} ({rec.get('ticker')})",
        "",
        _kv_block(
            "Position the model has taken",
            [
                ("side", f"{meta.get('side')} (rank {meta.get('rank')} of the {meta.get('region')} book)"),
                ("model score", _fmt(meta.get("prediction"), ".4f")),
                ("model's own factor commentary", meta.get("model_commentary") or "—"),
                ("region / sector / industry",
                 f"{meta.get('region')} / {meta.get('sector') or '—'} / {meta.get('industry') or '—'}"),
                ("market cap", f"{_fmt(meta.get('market_cap'), ',.0f')} {cur}"),
                ("holding week", f"{rec.get('as_of_date')} → {rec.get('target_date')}"),
            ],
        ),
        _kv_block(
            "Most recent reported earnings",
            [
                ("date", le.get("date") or "—"),
                ("EPS actual vs estimate",
                 f"{_fmt(le.get('eps_actual'))} vs {_fmt(le.get('eps_estimated'))} "
                 f"({_fmt(le.get('eps_surprise_pct'), '+.1f')}%)"),
                ("revenue actual vs estimate",
                 f"{_fmt(le.get('revenue_actual'), ',.0f')} vs {_fmt(le.get('revenue_estimated'), ',.0f')} "
                 f"({_fmt(le.get('revenue_surprise_pct'), '+.1f')}%)"),
            ],
        ),
        _kv_block(
            "Next earnings and forward expectation",
            [
                ("date", f"{ne.get('date') or '—'} ({_fmt(ne.get('days_until'), '.0f')} days away)"),
                ("lands inside the holding week", "YES — unhedged event risk" if ne.get("in_holding_week") else "no"),
                ("consensus EPS for that quarter", _fmt(ne.get("eps_estimated"))),
                ("consensus revenue for that quarter", _fmt(ne.get("revenue_estimated"), ",.0f")),
                ("next-FY consensus EPS",
                 f"{_fmt(est.get('eps_avg'))} (n={_fmt(est.get('n_analysts'), '.0f')}, FY to {est.get('date') or '—'})"),
            ],
        ),
        _kv_block(
            "Valuation (TTM)",
            [
                ("P/E", _fmt(val.get("pe_ttm"))),
                ("P/B", _fmt(val.get("pb_ttm"))),
                ("EV/EBITDA", _fmt(val.get("ev_ebitda_ttm"))),
                ("EV/sales", _fmt(val.get("ev_sales_ttm"))),
                ("EV/FCF", _fmt(val.get("ev_fcf_ttm"))),
                ("net debt / EBITDA", _fmt(val.get("net_debt_ebitda_ttm"))),
                ("gross margin %", _fmt(val.get("gross_margin_ttm"))),
                ("net margin %", _fmt(val.get("net_margin_ttm"))),
                ("ROE %", _fmt(val.get("roe_ttm"))),
            ],
        ),
        _kv_block(
            "Analyst consensus",
            [("status", cons.get("reason") or "—")]
            if not cons.get("available")
            else [
                ("rating", f"{cons.get('label')} "
                           f"(strong buy {cons.get('strong_buy')}, buy {cons.get('buy')}, "
                           f"hold {cons.get('hold')}, sell {cons.get('sell')}, "
                           f"strong sell {cons.get('strong_sell')})"),
                ("avg price target, last month", _fmt(cons.get("avg_pt_last_month"))),
                ("target vs last close %", _fmt(cons.get("pt_vs_price_pct"), "+.1f")),
            ],
        ),
        _kv_block(
            "Technicals (computed upstream — narrate, do not recompute)",
            [
                ("VERDICT", f"{tech.get('verdict')} — {tech.get('verdict_label')}"),
                ("why", "; ".join(tech.get("verdict_reasons") or []) or "—"),
                ("agreement with the model's side", tech.get("alignment") or "—"),
                ("flags", ", ".join(tech.get("flags") or []) or "none"),
                ("last close", f"{_fmt(tech.get('last_close'))} {cur} (as of {tech.get('asof') or '—'})"),
                ("RSI(14)", _fmt(tech.get("rsi14"), ".1f")),
                ("vs SMA20 / SMA50 / SMA200 %",
                 f"{_fmt(tech.get('px_vs_sma20_pct'), '+.1f')} / "
                 f"{_fmt(tech.get('px_vs_sma50_pct'), '+.1f')} / "
                 f"{_fmt(tech.get('px_vs_sma200_pct'), '+.1f')}"),
                ("return 1w / 1m / 3m %",
                 f"{_fmt(tech.get('ret_1w_pct'), '+.1f')} / "
                 f"{_fmt(tech.get('ret_1m_pct'), '+.1f')} / "
                 f"{_fmt(tech.get('ret_3m_pct'), '+.1f')}"),
                ("position in 52w range",
                 f"{_fmt((tech.get('pct_of_52w_range') or 0) * 100, '.0f')}% "
                 f"(low {_fmt(tech.get('range_52w_low'))}, high {_fmt(tech.get('range_52w_high'))})"
                 if tech.get("pct_of_52w_range") is not None else "—"),
                ("annualized 20d vol %", _fmt(tech.get("vol_20d_ann_pct"), ".0f")),
                ("move since the model scored it %", _fmt(tech.get("move_since_signal_pct"), "+.1f")),
            ],
        ),
    ]

    arts = news.get("articles") or []
    out.append(f"### News, last {news.get('window_days', 30)} days "
               f"(source: {news.get('source', 'none')})")
    if news.get("answer"):
        out.append(f"Search-engine summary: {news['answer']}")
    if not arts:
        out.append("- NO ARTICLES FOUND. The theme cannot be established from "
                   "available sources — say so rather than inferring one.")
    for a in arts:
        out.append(
            f"- [{a.get('published') or 'undated'}] {a.get('title')} "
            f"({a.get('publisher') or '?'})\n  URL: {a.get('url')}\n  {a.get('snippet') or ''}"
        )
    if rec.get("errors"):
        out.append("\n### Data gaps\n" + "\n".join(f"- {e}" for e in rec["errors"]))
    return "\n".join(out) + "\n"


def build_factsheet_pack(artifact: dict) -> str:
    """Every covered name's fact sheet in one markdown document.

    This is what the `session` backend hands to the running Claude Code session.
    """
    head = [
        f"# Deep-dive fact sheets — {artifact['signal_day']} "
        f"{artifact['as_of_date']} → {artifact['target_date']}",
        "",
        f"{len(artifact['tickers'])} names. Write one JSON brief per ticker, "
        "following the contract in `src/research/prompts/deepdive.md`.",
        "",
        "Emit a single JSON object mapping ticker -> brief, save it to a file, "
        "then merge it with:",
        "",
        "```",
        f"python -m src.research merge-narratives --signal-day {artifact['signal_day']} "
        f"--target-date {artifact['target_date']} --file <path.json>",
        "```",
        "",
        "---",
        "",
    ]
    body = []
    for ticker, rec in artifact["tickers"].items():
        rec = {**rec, "ticker": ticker,
               "as_of_date": artifact["as_of_date"], "target_date": artifact["target_date"]}
        body.append(build_factsheet(rec))
        body.append("\n---\n")
    return "\n".join(head + body)


# --- parsing and validation ---------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_narrative(text: str) -> tuple[dict | None, str | None]:
    """Lenient parse, strict validate.

    Models wrap JSON in fences and prefix it with a sentence often enough that
    it is worth recovering from, but the *shape* is not negotiable — a brief
    missing `technical_read` would render as a blank panel.
    """
    if not text or not text.strip():
        return None, "empty response"
    cleaned = _FENCE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None, "no JSON object found in response"
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, f"expected a JSON object, got {type(obj).__name__}"
    return validate_narrative(obj)


def validate_narrative(obj: dict) -> tuple[dict | None, str | None]:
    """Coerce to the contract, or reject with a reason."""
    missing = [k for k in REQUIRED_STR_KEYS if not isinstance(obj.get(k), str) or not obj[k].strip()]
    if missing:
        return None, f"missing or non-string keys: {', '.join(missing)}"

    out = {k: obj[k].strip() for k in REQUIRED_STR_KEYS}

    words = out["thesis"].split()
    if len(words) > MAX_THESIS_WORDS:
        out["thesis"] = " ".join(words[:MAX_THESIS_WORDS]) + "…"

    for key, limit in LIST_KEY_LIMITS.items():
        raw = obj.get(key) or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return None, f"{key} must be a list, got {type(raw).__name__}"
        out[key] = [str(x).strip() for x in raw if str(x).strip()][:limit]

    for key, allowed in ENUMS.items():
        val = str(obj.get(key, "")).strip().lower()
        out[key] = val if val in allowed else ("partly" if key == "model_agreement" else "low")
    return out, None


# --- the metered backend ------------------------------------------------------


def generate_via_api(rec: dict, model: str, max_tokens: int = 800) -> tuple[dict | None, dict, str | None]:
    """One brief through `src.reporting.llm`. Returns (narrative, usage, error).

    Retries once on a parse failure with an explicit nudge; a second failure
    leaves `narrative` null and the page degrades to the Tavily summary plus the
    computed verdict. Prompt caching is deliberately not used — the system
    prompt is well under `llm.ANTHROPIC_CACHE_MIN_TOKENS`, so the marker would
    be billed without ever producing a hit.
    """
    from src.reporting import llm

    system, user = system_prompt(), build_factsheet(rec)
    total = {"input_tokens": 0, "output_tokens": 0}
    last_err = None

    for attempt in range(2):
        prompt = user if attempt == 0 else (
            user + "\n\nYour previous reply was not valid JSON. Return ONLY the "
            "JSON object described above, with no surrounding prose or fence."
        )
        try:
            text, usage = llm.call(model, system, prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            return None, total, f"llm: {type(exc).__name__}: {exc}"
        total["input_tokens"] += usage.get("input_tokens", 0)
        total["output_tokens"] += usage.get("output_tokens", 0)
        narrative, last_err = parse_narrative(text)
        if narrative is not None:
            return narrative, total, None
    return None, total, f"llm: {last_err}"


def merge_narratives(artifact: dict, narratives: dict) -> tuple[int, list[str]]:
    """Validate a ticker -> brief mapping and write it onto the artifact."""
    merged, problems = 0, []
    for ticker, obj in (narratives or {}).items():
        if ticker not in artifact.get("tickers", {}):
            problems.append(f"{ticker}: not in this artifact")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{ticker}: expected an object, got {type(obj).__name__}")
            continue
        validated, err = validate_narrative(obj)
        if validated is None:
            problems.append(f"{ticker}: {err}")
            continue
        artifact["tickers"][ticker]["narrative"] = validated
        merged += 1
    covered = len(artifact.get("tickers", {}))
    for ticker in artifact.get("tickers", {}):
        if artifact["tickers"][ticker].get("narrative") is None and ticker not in (narratives or {}):
            problems.append(f"{ticker}: no brief supplied")
    artifact.setdefault("provenance", {})["n_narratives"] = sum(
        1 for r in artifact.get("tickers", {}).values() if r.get("narrative")
    )
    artifact["provenance"]["narrative_coverage"] = (
        f"{artifact['provenance']['n_narratives']}/{covered}"
    )
    return merged, problems
