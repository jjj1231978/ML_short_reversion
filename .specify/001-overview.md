# 001 — Overview

## Why

Short-horizon mean reversion in equities is well documented but hard to monetize: high
turnover and fast alpha decay eat the gross signal. Société Générale's April 2025 paper
("Can ML help us design a better stock mean reversion strategy?") shows that a
gradient-boosted-trees model — given an unconstrained library of ~86 cross-sectional
factors — naturally rediscovers reversal as the dominant weekly signal, but with
**lower turnover, lower seasonality, and better drawdown profile** than a hand-built
1-week reversal strategy or an earnings-filtered version of it. Their reported
net-of-cost numbers (Global L/S, 2006-2025): 9.7% ann. return, IR 1.6, max DD 12.3%.

This project replicates and extends that work in our own infrastructure so we can:

- Verify the headline findings on independently sourced data.
- Use it as a research substrate for adding new factors / different tree models /
  alternative target horizons.
- Maintain a permanent ML-research codebase that doesn't depend on SG's paywalled
  Quant Data API.

## Core hypothesis

Given a broad cross-sectional factor library and a forward 1-week return target, an
unconstrained ML model will:

1. Rank short-term price reversal (R1W) as the most important feature (top SHAP).
2. Rank short-horizon earnings revisions (UPDOWN1W) as the second most important.
3. Generate a long-short portfolio with **higher information ratio** and **lower
   turnover** than either a pure R1W reversal strategy or an EPS-revision-filtered
   reversal baseline.
4. Show alpha decay with execution lag (worse with delay) and a weekday signature
   (Thursday signals > Monday signals), driven by the timing of macro / earnings
   releases.

### Phase 1 verdict (2026-05-15)

| Claim | Status |
|---|---|
| (1) R1W is #1 SHAP | **Not confirmed.** Top features are VOL12M, VOL6M, R12M1M; R1W is #4. |
| (2) UPDOWN1W is #2 | **Untestable on current data.** True EPS-revision UPDOWN1W not yet sourced. FMP grade-event proxy `UPDOWN1W_RATINGS` ranks #23 — strong evidence the proxy is not a substitute. |
| (3) ML IR > R1W baseline | **Marginal.** Ensemble IR 1.00 vs baseline ≈ 1.10; ML's edge is in turnover (0.51 vs ~40x) and drawdown (-11% vs much worse), not IR. |
| (4) Alpha decay + weekday signature | **Both confirmed.** Monotone lag 12.7% → 4.7%; monotone Mon→Fri 22% → 35%. |

(1) being off makes sourcing the true UPDOWN1W feature the load-bearing Phase 2
experiment — it directly tests whether the paper's claim depends on the
EPS-revision feature being present in the library. See [`007`](007-implementation-plan.md)
block E.

## Phased scope

This spec is **deliberately phased.** Phase 1 ships a working end-to-end version on
a narrow universe and limited factor set. Phase 2 expands toward paper parity once
Phase 1 is validated.

### Phase 1 (shipped 2026-05-15)

| Dimension | Phase 1 | Paper |
|---|---|---|
| Universe | US S&P 500 (PIT) + UK FTSE 100 + CA TSX 60 (snapshot), ex-financials | Global developed (US + EU + JP), large/mid cap |
| Backtest window | 2008-01-30 → 2026-05-13 (~18 yrs, 955 weeks) | 2006-2025 (~19 yrs) |
| Factor count | 24 (price + risk + value + profitability/quality + FMP-grades proxy + macro/regime passthrough) | 86 (80 base + 6 weekly add-ons) |
| Earnings revisions | FMP `/grades` proxy only — true EPS-revision UPDOWN1W absent | UPDOWN1W is the #2 SHAP feature in the paper |
| Train / val window | 78 / 26 weeks (held-over Phase-1 default; bump to 520/104 deferred — see [`007`](007-implementation-plan.md) task H.10) | 520 / 104 weeks |
| Quintile peer group | universe-wide | region × industry |
| Universe membership | US PIT (FMP back to 1957); UK/CA snapshot-only (survivorship-biased — accepted FMP limitation) | point-in-time |
| Model | ensemble (XGB + LGBM + RF + MLP), equal-weight | Gradient boosted trees |
| Net IR (US/UK/CA L/S) | ensemble 1.00 / best member RF 1.13 / R1W baseline ≈ 1.10 | Global L/S 1.6 |

Phase 1 lands **most** of the paper's structural choices (PIT US universe, long
history, multi-region scope, ensemble model, neutralization, ADV-scaled basket,
exec lag, SHAP diagnostics). The remaining gaps that Phase 2 closes are the
factor-library size (24 → 86), the EPS-revision factor specifically, peer
grouping (universe-wide → region × industry), and full multi-region (no EU/JP
yet).

### Phase 2 (paper replication)

- Earnings-revision data source (e.g. IBES via WRDS, FMP, or Refinitiv) wired into
  the factor pipeline; implements UPDOWN1W, SUE1W, SUE3, SUE6.
- Longer price history source (e.g. CRSP, paid Databento entitlement, or yfinance
  fallback for pre-2023) so the 520/104 paper window fits.
- Point-in-time universe membership.
- Region × industry quintile peer groups.
- Multi-region support (EU + JP added).
- Factor expansion toward 86.

Phase 2 timing is gated on Phase 1 validation passing the checks in
[`004-constraints.md`](004-constraints.md).

### Data-source caveats (FMP migration, 2026-Q2)

- **US (SP500) universe is point-in-time** — derived from FMP's
  `/stable/historical-sp500-constituent` event stream (back to 1957).
- **UK (FTSE 100) and Canada (TSX 60) universes are current-snapshot only and
  therefore survivorship-biased.** FMP does not expose historical FTSE 100 /
  TSX 60 membership. Approximated via `/stable/company-screener` filtered by
  exchange and market-cap floor. Accepted as a known limitation of the data
  source; no engineering workaround.
- **Currency mismatch in PE / PB for non-US tickers.** FMP reports income
  statements in `reportedCurrency` (sometimes USD even for LSE-listed firms,
  e.g. HSBC), while prices are in the exchange's local currency (LSE in GBp,
  TSX in CAD). The within-(region × industry) z-score in the neutralization
  step partially neutralizes this; full FX handling deferred. ROE and GPOA are
  unitless ratios and unaffected.

## Out of scope (both phases)

- Intraday signals or sub-daily data.
- Non-equity asset classes.
- Live trading / order management — research backtest only.
- Risk model (Barra-style) overlay; neutralization is industry-median only.

## Open questions

| Q | Owner | Notes |
|---|---|---|
| Earnings-revision data source for Phase 2 | user | Cost vs coverage trade-off; IBES via WRDS is the gold standard but expensive. |
| Long-history price source for Phase 2 | user | Faithful 520-week train requires pre-2023 history. CRSP, paid Databento back-pull, or yfinance bridge. |
| Acceptable IR threshold for Phase 1 to pass validation | user | Paper net-of-cost US IR is 1.0. With only 3 years of data, expect noisier estimates. |
