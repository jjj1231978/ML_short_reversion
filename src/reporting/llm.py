"""Provider-agnostic LLM call dispatch for Anthropic and OpenAI.

Loads API keys from `.env` (gitignored) at import time. Dispatches based on the
model ID prefix: `claude-*` → Anthropic, `gpt-*` / `o1-*` / `o3-*` → OpenAI.

Pricing constants are approximate; see `meta.json` after a run for actuals.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

log = logging.getLogger(__name__)


# USD per million tokens (input, output). Approximate; update if pricing shifts.
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

# Anthropic prompt caching: cache writes are billed at 1.25x the input rate
# (5-minute TTL), cache reads at 0.1x.
ANTHROPIC_CACHE_WRITE_MULT = 1.25
ANTHROPIC_CACHE_READ_MULT = 0.10

# Approximate chars per token for cost pre-estimation. Both providers tokenize
# English text at roughly 4 chars/token; this is good enough for budget guard.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough char-based token estimate. Good enough for cost guard, not billing."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_cost_usd(model: str, input_text: str, max_output_tokens: int) -> float:
    """Upper-bound cost estimate assuming output saturates `max_output_tokens`."""
    if model not in PRICING:
        log.warning(f"No pricing entry for {model!r}; assuming opus-tier rates")
        in_rate, out_rate = (15.00, 75.00)
    else:
        in_rate, out_rate = PRICING[model]
    in_tokens = estimate_tokens(input_text)
    return (in_tokens * in_rate + max_output_tokens * out_rate) / 1_000_000


def actual_cost_usd(model: str, usage: dict) -> float:
    """Cost from real usage counts returned by the provider.

    Accounts for Anthropic prompt-caching pricing: cache writes are 1.25x the
    standard input rate, cache reads are 0.1x. OpenAI's cache discount is
    applied transparently in the API and is not surfaced in usage counts, so
    we just charge input_tokens at the standard rate there.
    """
    in_rate, out_rate = PRICING.get(model, (15.00, 75.00))
    in_t = usage.get("input_tokens", 0)
    out_t = usage.get("output_tokens", 0)
    cw_t = usage.get("cache_creation_input_tokens", 0)
    cr_t = usage.get("cache_read_input_tokens", 0)
    cost = (
        in_t * in_rate
        + out_t * out_rate
        + cw_t * in_rate * ANTHROPIC_CACHE_WRITE_MULT
        + cr_t * in_rate * ANTHROPIC_CACHE_READ_MULT
    )
    return cost / 1_000_000


def is_anthropic(model: str) -> bool:
    return model.startswith("claude-")


def is_openai(model: str) -> bool:
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def call(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 8000,
    cached_prefix: str | None = None,
) -> tuple[str, dict]:
    """Call an LLM and return (response_text, usage_dict).

    Args:
        model: Provider model id, e.g. "claude-haiku-4-5" or "gpt-4o-mini".
        system: System prompt (per-agent, not cached).
        user: User message (the dynamic per-call content — for critique that's
            the draft; for synthesis, draft + critique).
        max_tokens: Output cap.
        cached_prefix: For Anthropic only — text prepended as the first system
            block with cache_control. Must be identical across calls within a
            run for cache hits to materialize. Sent ahead of `system` so the
            cache key (= prefix) matches across agents that share this prefix.

    Returns:
        (text, usage) where usage has keys: input_tokens, output_tokens,
        cache_read_input_tokens, cache_creation_input_tokens, elapsed_seconds,
        model.
    """
    if is_anthropic(model):
        return _call_anthropic(model, system, user, max_tokens, cached_prefix)
    if is_openai(model):
        return _call_openai(model, system, user, max_tokens)
    raise ValueError(f"Unknown model id: {model!r}. Use claude-* or gpt-*/o*.")


# Anthropic's prompt caching has a minimum prefix size; below this the API
# accepts the cache_control marker but does not actually cache anything.
# Conservatively set above the documented Haiku minimum (~1K-2K tokens).
ANTHROPIC_CACHE_MIN_TOKENS = 2048


def _call_anthropic(
    model: str, system: str, user: str, max_tokens: int, cached_prefix: str | None
) -> tuple[str, dict]:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY missing. Add it to .env at the repo root."
        )

    client = anthropic.Anthropic()

    if cached_prefix and estimate_tokens(cached_prefix) >= ANTHROPIC_CACHE_MIN_TOKENS:
        system_arg = [
            {
                "type": "text",
                "text": cached_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": system},
        ]
    else:
        system_arg = system

    t0 = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_arg,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0

    text = "".join(b.text for b in msg.content if b.type == "text")
    usage = {
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(
            msg.usage, "cache_creation_input_tokens", 0
        ) or 0,
        "elapsed_seconds": round(elapsed, 2),
        "model": model,
    }
    return text, usage


def _call_openai(
    model: str, system: str, user: str, max_tokens: int
) -> tuple[str, dict]:
    import openai

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY missing. Add it to .env at the repo root."
        )

    client = openai.OpenAI()

    # o-series reasoning models use max_completion_tokens instead of max_tokens.
    extra = {}
    if model.startswith(("o1-", "o3-", "o4-")):
        extra["max_completion_tokens"] = max_tokens
    else:
        extra["max_tokens"] = max_tokens

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **extra,
    )
    elapsed = time.time() - t0

    text = resp.choices[0].message.content or ""
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "elapsed_seconds": round(elapsed, 2),
        "model": model,
    }
    return text, usage
