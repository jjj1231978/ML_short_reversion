"""Tests for narrative parsing, validation and merging (src/research/narrative.py).

The parser is deliberately lenient about *packaging* (fences, stray prose) and
strict about *shape* — a brief missing `technical_read` would render as a blank
panel on the page, so it must be rejected rather than half-accepted.
"""

import json

import pytest

from src.research import narrative as nar

GOOD = {
    "thesis": "Beat and raised, but the stock is stretched two sigma above its 20-day mean.",
    "theme": "AI optics demand pulling datacom revenue higher.",
    "fundamental_read": "EPS of 1.74 beat the 1.62 estimate by 7.4%.",
    "technical_read": "RSI 73 with the verdict flagging reversal risk into the holding week.",
    "headline_risks": ["Guidance assumes no China export change, per https://example.com/a"],
    "catalysts": ["Q1 FY27 earnings 2026-11-04"],
    "model_agreement": "disagrees",
    "confidence": "high",
    "sources_used": ["https://example.com/a", "fmp:earnings"],
}


def test_plain_json():
    out, err = nar.parse_narrative(json.dumps(GOOD))
    assert err is None
    assert out["thesis"].startswith("Beat and raised")
    assert out["model_agreement"] == "disagrees"


def test_fenced_json():
    out, err = nar.parse_narrative("```json\n" + json.dumps(GOOD) + "\n```")
    assert err is None and out["theme"] == GOOD["theme"]


def test_bare_fence_without_language():
    out, err = nar.parse_narrative("```\n" + json.dumps(GOOD) + "\n```")
    assert err is None and out is not None


def test_leading_and_trailing_prose_is_recovered():
    text = "Here is the brief you asked for:\n" + json.dumps(GOOD) + "\nLet me know if you need more."
    out, err = nar.parse_narrative(text)
    assert err is None and out["confidence"] == "high"


def test_empty_response_rejected():
    for bad in ("", "   ", None):
        out, err = nar.parse_narrative(bad)
        assert out is None and err


def test_no_json_at_all_rejected():
    out, err = nar.parse_narrative("I could not find enough information about this company.")
    assert out is None
    assert "no JSON object" in err


def test_malformed_json_rejected():
    out, err = nar.parse_narrative('{"thesis": "x", "theme": }')
    assert out is None and err


def test_non_object_json_rejected():
    out, err = nar.parse_narrative('["thesis", "theme"]')
    assert out is None
    assert "object" in err


@pytest.mark.parametrize("missing", list(nar.REQUIRED_STR_KEYS))
def test_missing_required_key_rejected(missing):
    obj = {k: v for k, v in GOOD.items() if k != missing}
    out, err = nar.parse_narrative(json.dumps(obj))
    assert out is None
    assert missing in err


def test_blank_required_key_rejected():
    out, err = nar.parse_narrative(json.dumps({**GOOD, "technical_read": "   "}))
    assert out is None and "technical_read" in err


def test_wrong_type_for_required_key_rejected():
    out, err = nar.parse_narrative(json.dumps({**GOOD, "thesis": 42}))
    assert out is None and "thesis" in err


def test_list_keys_are_truncated_to_the_contract():
    obj = {**GOOD, "headline_risks": [f"risk {i}" for i in range(10)]}
    out, _ = nar.parse_narrative(json.dumps(obj))
    assert len(out["headline_risks"]) == nar.LIST_KEY_LIMITS["headline_risks"]


def test_scalar_list_key_is_promoted_to_a_list():
    out, err = nar.parse_narrative(json.dumps({**GOOD, "catalysts": "earnings 2026-11-04"}))
    assert err is None and out["catalysts"] == ["earnings 2026-11-04"]


def test_non_list_non_str_list_key_rejected():
    out, err = nar.parse_narrative(json.dumps({**GOOD, "headline_risks": {"a": 1}}))
    assert out is None and "headline_risks" in err


def test_missing_list_keys_default_to_empty():
    obj = {k: v for k, v in GOOD.items() if k not in ("headline_risks", "catalysts")}
    out, err = nar.parse_narrative(json.dumps(obj))
    assert err is None
    assert out["headline_risks"] == [] and out["catalysts"] == []


def test_blank_list_entries_dropped():
    out, _ = nar.parse_narrative(json.dumps({**GOOD, "catalysts": ["", "  ", "real one"]}))
    assert out["catalysts"] == ["real one"]


def test_thesis_is_truncated_not_rejected():
    long_thesis = " ".join(f"word{i}" for i in range(60))
    out, err = nar.parse_narrative(json.dumps({**GOOD, "thesis": long_thesis}))
    assert err is None
    assert len(out["thesis"].split()) == nar.MAX_THESIS_WORDS + 0
    assert out["thesis"].endswith("…")


@pytest.mark.parametrize(
    "key,bad,expected",
    [("model_agreement", "maybe", "partly"), ("confidence", "extremely", "low")],
)
def test_out_of_vocabulary_enum_falls_back_conservatively(key, bad, expected):
    out, err = nar.parse_narrative(json.dumps({**GOOD, key: bad}))
    assert err is None and out[key] == expected


def test_enum_case_is_normalized():
    out, _ = nar.parse_narrative(json.dumps({**GOOD, "confidence": "HIGH"}))
    assert out["confidence"] == "high"


def test_missing_enum_defaults():
    obj = {k: v for k, v in GOOD.items() if k not in ("model_agreement", "confidence")}
    out, err = nar.parse_narrative(json.dumps(obj))
    assert err is None
    assert out["model_agreement"] == "partly" and out["confidence"] == "low"


# --- merge --------------------------------------------------------------------


def _artifact():
    return {
        "signal_day": "THU", "as_of_date": "2026-08-13", "target_date": "2026-08-20",
        "tickers": {
            "COHR": {"meta": {"side": "LONG"}, "narrative": None},
            "KLR.L": {"meta": {"side": "SHORT"}, "narrative": None},
        },
    }


def test_merge_writes_validated_briefs():
    art = _artifact()
    merged, problems = nar.merge_narratives(art, {"COHR": GOOD, "KLR.L": GOOD})
    assert merged == 2 and problems == []
    assert art["tickers"]["COHR"]["narrative"]["confidence"] == "high"
    assert art["provenance"]["narrative_coverage"] == "2/2"


def test_merge_reports_unknown_ticker():
    art = _artifact()
    merged, problems = nar.merge_narratives(art, {"NOPE": GOOD})
    assert merged == 0
    assert any("NOPE" in p and "not in this artifact" in p for p in problems)


def test_merge_rejects_an_invalid_brief_without_losing_the_valid_ones():
    art = _artifact()
    merged, problems = nar.merge_narratives(
        art, {"COHR": GOOD, "KLR.L": {k: v for k, v in GOOD.items() if k != "theme"}}
    )
    assert merged == 1
    assert art["tickers"]["COHR"]["narrative"] is not None
    assert art["tickers"]["KLR.L"]["narrative"] is None
    assert any("KLR.L" in p and "theme" in p for p in problems)


def test_merge_flags_names_with_no_brief_supplied():
    art = _artifact()
    _, problems = nar.merge_narratives(art, {"COHR": GOOD})
    assert any("KLR.L" in p and "no brief supplied" in p for p in problems)
    assert art["provenance"]["narrative_coverage"] == "1/2"


def test_merge_handles_a_non_object_brief():
    art = _artifact()
    merged, problems = nar.merge_narratives(art, {"COHR": "just a string"})
    assert merged == 0 and any("COHR" in p for p in problems)


def test_merge_on_empty_input_is_a_no_op():
    art = _artifact()
    merged, problems = nar.merge_narratives(art, {})
    assert merged == 0
    assert art["provenance"]["n_narratives"] == 0
