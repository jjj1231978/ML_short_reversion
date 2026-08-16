"""Tavily news search — the news and theme layer for every region.

FMP's `news/stock` endpoint returns an empty list for `.L` and `.TO` symbols
(verified against the live API), so two thirds of this book has no vendor news
at all. Tavily covers all three regions from the open web, and its
`include_answer` field returns a one-paragraph synthesis that doubles as the
fallback summary when the narrative layer is unavailable.

The API key lives in `.env` as `TAVILY_API_KEY`. Responses are cached for a day
under the same deep-dive cache tree as the FMP calls, so re-runs are free.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.research.fmp_deep import _cache_path, _read_cache, _write_cache

load_dotenv(PROJECT_ROOT / ".env")

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_DAYS = 30
DEFAULT_MAX_RESULTS = 6
TTL_DAYS = 1.0

# Snippets dominate the artifact's size — enough to establish what a story is
# about and let the reader decide whether to open the link, no more.
SNIPPET_CHARS = 350
MAX_ARTICLES = 8


def has_key() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def build_query(ticker: str, company_name: str | None, region: str) -> str:
    """A query specific enough to avoid ticker collisions.

    Bare tickers are ambiguous on the open web ('A', 'BB'), so lead with the
    company name and name the exchange for non-US listings.
    """
    exchange = {"UK": "London Stock Exchange", "CA": "Toronto Stock Exchange"}.get(region, "")
    name = company_name or ticker
    parts = [name, f"({ticker})", exchange, "stock news earnings guidance outlook"]
    return " ".join(p for p in parts if p)


def _post(payload: dict) -> dict:
    """Call Tavily via the official SDK, falling back to a plain POST.

    Both imports are lazy so the Streamlit Space can render cached artifacts
    without either package installed.
    """
    key = os.environ["TAVILY_API_KEY"]
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=key).search(**payload)
    except ImportError:
        import requests

        resp = requests.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def search_news(
    ticker: str,
    company_name: str | None = None,
    region: str = "US",
    days: int = DEFAULT_DAYS,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict:
    """Recent news for one name.

    Always returns a dict with the same shape — `articles` is `[]` and `error`
    is set when the search fails, so the narrative layer can be told plainly
    that no news was found rather than being left to invent some.
    """
    query = build_query(ticker, company_name, region)
    base = {
        "source": "tavily",
        "query": query,
        "window_days": days,
        "answer": None,
        "articles": [],
        "error": None,
    }

    if not has_key():
        base["error"] = "TAVILY_API_KEY not set"
        return base

    params = {
        "query": query,
        "topic": "news",
        "days": days,
        "max_results": max_results,
        "include_answer": "basic",
        "search_depth": "basic",
    }
    fp = _cache_path("tavily/news", {"symbol": ticker, **params}, cache_dir)

    raw = None if force else _read_cache(fp, TTL_DAYS)
    if raw is None:
        try:
            raw = _post(params)
            _write_cache(fp, raw)
        except Exception as exc:  # noqa: BLE001 — news is best-effort
            log.warning(f"tavily {ticker}: {exc}")
            base["error"] = f"tavily: {type(exc).__name__}: {exc}"
            return base

    base["answer"] = raw.get("answer")
    base["articles"] = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "published": r.get("published_date"),
            "publisher": _publisher(r.get("url")),
            "score": round(r["score"], 3) if isinstance(r.get("score"), (int, float)) else None,
            "snippet": (r.get("content") or "")[:SNIPPET_CHARS],
        }
        for r in (raw.get("results") or [])
    ]
    return base


def _publisher(url: str | None) -> str | None:
    """Bare domain, for display next to each headline."""
    if not url:
        return None
    try:
        from urllib.parse import urlparse

        return (urlparse(url).netloc or "").removeprefix("www.") or None
    except Exception:
        return None


def merge_fmp_news(news: dict, fmp_rows: list | None, max_items: int = 6) -> dict:
    """Fold FMP's vendor news (US only) in alongside the web results.

    Vendor rows are listed first — they are symbol-tagged at source, so they
    cannot be about a different company that happens to share a ticker.
    """
    if not fmp_rows:
        return news
    vendor = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "published": r.get("publishedDate"),
            "publisher": r.get("publisher") or r.get("site"),
            "score": None,
            "snippet": (r.get("text") or "")[:SNIPPET_CHARS],
        }
        for r in fmp_rows[:max_items]
    ]
    seen = {a["url"] for a in vendor if a.get("url")}
    web = [a for a in news["articles"] if a.get("url") not in seen]
    news["articles"] = (vendor + web)[:MAX_ARTICLES]
    news["source"] = "fmp+tavily" if news.get("answer") or web else "fmp"
    return news
