# 007 — Implementation Plan

Status of each component, gaps vs paper / spec, and the ordered work list.
Cross-references the requirements in [`002`](002-requirements.md), procedures in
[`003`](003-procedures.md), and constraints in [`004`](004-constraints.md).

Last refresh: 2026-05-15. Phase 1 has shipped — every item in the original A/B/C/C+
work lists is in code. This doc now tracks Phase 2 + observations from the Phase 1
results that should shape Phase 2 priorities.

## Status by component (as of 2026-05-15)

| Component | Status | Notes |
|---|---|---|
| Universe — US S&P 500 PIT | done | FMP `/stable/historical-sp500-constituent` event stream (back to 1957) |
| Universe — UK FTSE 100 + CA TSX 60 | done (survivorship-biased) | FMP has no PIT history; current snapshot only |
| Per-week eligibility mask (PIT × price × ADV) | done | `build_eligibility_mask` applied before quintile assignment |
| ADV filter | done | Per-region native-currency thresholds (USD / GBp / CAD) |
| Price fetch | done | Multi-source: Databento (intraday-grade) + FMP (long history); current backtest covers 2008-01-30 → 2026-05-13 |
| Fundamentals fetch | done | SimFin + FMP, 90-day publication lag |
| Analyst-grade events (UPDOWN1W proxy) | done | FMP `/grades`; named `UPDOWN1W_RATINGS` to stay honest vs paper's EPS-revision UPDOWN1W |
| Macro block | done | FMP series + HMM regime posteriors as passthrough features |
| Factor library | partial — 24 of 86 | Price/risk/momentum/value/profitability/quality + analyst-grade proxy + macro/regime; see [`002`](002-requirements.md) §3 |
| Factor: IREV1W (beta-adjusted residual) | done | `src/features/factors.py:irev1w` |
| Factor: BETA6M, IVOL, IMOM (market-relative) | done | All consume aligned `market_close` |
| Factor: UPDOWN1W (EPS revisions, paper's #2) | **not present** | Requires IBES / Refinitiv / similar; FMP grade events ≠ EPS estimates |
| Factor: SUE1W, SUE3, SUE6 | not present | Same data-source gap |
| Neutralization pipeline | done | Stacked cross-sectional: winsorize → iter z-score → ±3 cap → industry-median → drop > 10 missing + zero-fill |
| Passthrough cols (macro/regime) | done | Bypass cross-sectional collapse; participate in missing-budget |
| Wednesday resampling + cross-sectional z-scored 1W forward target | done | |
| Rolling train/val/test | done | Window sizes 78/26/1 in `configs/default.yaml`; data now supports paper's 520/104 — config bump deferred to validation step |
| Retrain cadence | done | every 12 weeks |
| Ensemble | done | XGBoost + LightGBM + RandomForest + MLP, equal-weighted prediction average |
| Quintile assignment | **universe-wide** | Paper uses region × GICS-industry peer groups |
| L/S portfolio + ADV-scaled basket weights | done | `min(1, ADV / threshold)` then renormalize; per-region thresholds |
| Cost model | done | 1.5 bps/side, applied via turnover; 1-day execution lag enforced (`main.py` refuses lag<1) |
| Weekly return cap (tail-event guard) | done | Configurable per `backtest.weekly_return_cap` |
| SHAP fan-out + persisted parquet + PNGs | done | Per ensemble member (xgb/lgb/rf/mlp); MLP path uses `GradientExplainer` in standardized space |
| Alpha decay diagnostic | done | Lags 0-4 |
| Weekday effect diagnostic | done | 5 full retrains, persisted to `weekday_effect.parquet` |
| Baselines | partial | Sector-relative R1W reversal done; earnings-filtered R1W gated on true UPDOWN1W |
| Streamlit app | done | 7 pages: overview / predictions / SHAP / alpha decay / weekday / param explorer / report |
| Multi-agent report generation | done | `src/reporting/`: briefing → critique → synthesize, Anthropic/OpenAI providers, prompt caching |
| Tests | partial | Unit tests for factors, neutralization (stacked), ADV scaling, backtest, macro factors; no full-pipeline integration test |

## Phase 1 results — observations that should shape Phase 2

Headline metrics over 2008-01-30 → 2026-05-13 (955 weeks, net of 1.5 bps/side, 1-day lag):

| Model | Ann. return | Ann. vol | IR | Max DD | Turnover |
|---|---|---|---|---|---|
| Random Forest | 10.2% | 9.0% | **1.13** | -14.5% | 0.51 |
| XGBoost | 8.8% | 8.5% | 1.03 | -15.2% | 0.48 |
| MLP | 8.0% | 7.8% | 1.03 | -10.7% | 0.43 |
| LightGBM | 6.6% | 7.9% | 0.83 | -13.6% | 0.45 |
| Ensemble (avg) | 8.8% | 8.8% | 1.00 | -11.0% | 0.51 |
| R1W reversal baseline | 12.6% | 11.5% | ~1.10 | n/a | very high |
| Paper net Global L/S | 9.7% | n/a | **1.6** | -12.3% | n/a |

Findings that bear on Phase 2 work priorities:

1. **SHAP order disagrees with the paper.** Top factors are VOL12M, VOL6M, R12M1M, R1W (paper says R1W first, EPS revisions second). The hypothesis in [`001`](001-overview.md) §"Core hypothesis" point (1) is **not confirmed** — R1W is #4, not #1. Phase 2 sourcing of true UPDOWN1W is now the highest-leverage experiment: it directly tests whether the paper's claim hinges on having the EPS-revision feature in the library at all.
2. **`UPDOWN1W_RATINGS` (FMP grade-event proxy) ranks #23.** Strong evidence that analyst grade events are not a substitute for EPS estimate revisions. Don't expect this proxy to fill the gap.
3. **ML edge over the R1W baseline is in turnover + drawdown, not return.** Ensemble IR ≈ baseline IR; ensemble turnover 0.51 vs baseline ~40x weekly turnover; ensemble max DD -11% vs baseline (uncomputed but pure R1W is known to whipsaw harder).
4. **Per-model spread is wide.** RF (1.13) > XGB / MLP (1.03) > LGBM (0.83). The equal-weight ensemble drags toward LGBM. Phase 2: consider IR-weighted ensembling or dropping LGBM, but only after the bigger Phase 2 levers are evaluated (peer grouping change can move all four members).
5. **Alpha decay and weekday effect both validate the paper's framing** — monotone decay 12.7% → 4.7% across lags 0-4; monotone Mon→Fri ramp 22.1% → 35.4%.

## Phase 2 work list — ordered by expected information value

Earnings revisions go first because Phase 1 result (1) makes it the load-bearing
hypothesis test. The rest is ordered by a mix of cost-to-implement and expected
move in IR.

### E. Earnings revisions (high leverage)

1. **Source decision for EPS estimate revisions.** Open question; candidates:
   IBES via WRDS (gold-standard, expensive), Refinitiv I/B/E/S API, FMP's
   `/analyst-estimates` (cheaper, lower coverage). The decision belongs to the
   user; this task is a 1-page comparison memo in
   `.specify/008-earnings-revision-source.md` with coverage / latency / price /
   PIT-fidelity columns, recommending one.
2. **Implement true UPDOWN1W** — count of estimate-direction changes over the
   trailing 5 trading days, signed. Spec: [`002`](002-requirements.md) §3.
3. **Implement SUE1W, SUE3, SUE6** — Standardized Unexpected Earnings at 1w / 3m /
   6m windows. Same data source.
4. **Earnings-filtered R1W baseline** in `src/backtest/baselines.py` — paper's
   strategy (b). Persist to `baseline_r1w_earnings_filtered_returns.parquet` and
   expose in the Streamlit overview page.
5. **Re-run the full pipeline** and validate against [`001`](001-overview.md)
   "Core hypothesis" points (1)–(3). If R1W still ranks below low-vol factors
   with true UPDOWN1W in the library, that is a publishable negative result.

### F. Peer grouping (medium leverage, low cost)

6. **Region × GICS-industry quintile assignment** in
   `src/backtest/portfolio.py:assign_quintiles`. Currently groups only by date.
   Two changes:
   - Take a `peer_map: dict[ticker, str]` argument formed by `(region, industry)`.
   - `groupby([date, peer])` instead of `groupby(date)`.
   The neutralization pipeline already subtracts the (date, industry) median —
   this is the analogous change one layer downstream. Re-run, persist
   `model_comparison_peer_grouped.json` alongside the existing file so the
   delta is auditable.
7. **Region-aware neutralization** — currently the cross-section is global per
   date. Should be (date × region) per factor. Touches
   `neutralize_stacked` (add `region_map` arg, fold into groupby) and `main.py`
   (pass it). Test via `tests/test_neutralization_stacked.py` with a 2-region
   fixture.

### G. Factor expansion toward 86 (medium leverage, medium cost)

8. **Prioritize by paper SHAP table.** Paper's top-20 SHAP includes a few that
   are absent here: short-interest, days-to-cover, liquidity-adjusted IVOL,
   12M-skipping-7D momentum, and 4-5 quality-flavored ones. List in
   `.specify/009-factor-expansion-priority.md` with availability per data source.
9. **Implement, neutralize, re-train**. Likely 10-15 factors gettable from
   FMP/SimFin without new vendors; remaining ~35 gated on alternative data.

### H. Train/val window (low-cost; gated on validation results)

10. **Bump to 520/104** in `configs/default.yaml` now that history supports it.
    Compare against the 78/26 baseline. Expectation: lower variance in OOS IR
    week-to-week, similar mean. Persist a `model_comparison_long_window.json`.

### I. Region expansion (high cost, gated on Phase 2 E + F passing)

11. **Add EU + JP universes.** FMP coverage check first — if PIT membership is
    only US-grade, EU/JP will be survivorship-biased like UK/CA. Decide whether
    to accept that or block on a PIT vendor.

### J. Tests / hygiene (continuous)

12. **Full-pipeline integration test** — 50 tickers × 200 weeks of synthetic
    data; assert `metrics_summary.json` is produced with non-NaN IR and SHAP
    files exist.
13. **Test for region × industry peer grouping** (depends on task F.6).
14. **Test for true UPDOWN1W computation** (depends on task E.2).

## Documentation backlog

- `CLAUDE.md` still says Phase 1 is 2023-04 → today on ~3 years of history. **Stale**: the
  current backtest uses 2008-01-30 → 2026-05-13 (955 weeks). Update CLAUDE.md to match,
  and point the long-form story here.
- `001-overview.md` Phase-1-vs-paper table still lists "78 / 26 weeks" and "current
  Wikipedia (survivorship-biased)" — both have moved on. Refresh once the user confirms
  the Phase 2 priorities above.

## Open questions

| Q | Owner | Notes |
|---|---|---|
| EPS-revision data source | user | Drives task E.1. IBES/WRDS, Refinitiv, FMP estimates — comparison memo in 008. |
| Acceptable IR threshold to declare Phase 1 validated | user | Paper net US IR ≈ 1.0; current ensemble = 1.00 / best member = 1.13. Is this "passed" already, or gated on hypothesis (1) holding (R1W as #1 SHAP)? |
| EU/JP PIT membership source for Phase 2 | user | If unavailable on FMP, do we accept survivorship bias for those regions or block? |
| Ensemble weighting | author | Equal-weight is naive; IR-weighted is the obvious next step. Defer until Phase 2 E settles which factors are in. |
