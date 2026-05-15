# 004 — Constraints

Hard rules and biases. Each is paired with how the codebase enforces (or fails to
enforce) it. Validation rules are at the bottom.

## Tradeable-only reporting (foundational rule)

**Any statistic that headlines this project — IR, return, drawdown, turnover, per-quintile spread, net-of-cost performance — must be computable from a position sequence that could actually be traded.** A number that requires knowledge unavailable at trade time, or a fill price unreachable from the signal, is not a result; it is a leakage artifact.

Concrete tests, applied before any metric is reported as a project finding:

| Test | What to check |
|---|---|
| Signal-to-fill gap | Returns must be measured starting **at least one full close after** the close that produced the signal. Lag-0 (Wed-close signal → Wed-close fill) is leakage and must not appear in headline metrics. |
| Feature timestamps | No factor input may use data published, revised, or stamped after the signal close. Includes fundamental restatements (use `pub_lag_days`), analyst revisions (point-in-time only), index membership (date_added / date_removed). |
| Universe membership | The set of tickers eligible at week `t` must be derivable from data available at or before close `t`. No survivorship-biased back-projection of a current snapshot through historical periods (US PIT only; UK/CA snapshot is documented as a known bias). |
| Cost realism | A reported "net of cost" number must subtract the costs that would have been paid by the live trader: commission/half-spread bps, ADV-scaled position sizing, and (when sourced) borrow on shorts. |

Diagnostic sweeps (e.g., `alpha_decay_analysis` over lags 0–4) **may include** unrealistic operating points to anchor the decay curve, but each such point must be labeled as diagnostic, not reported as a project performance number. The headline run uses `execution_lag_days >= 1`.

When in doubt: ask "could a live trader, knowing only what was visible at the signal close, have produced this position sequence?" If the answer is no, the metric does not ship.

## Look-ahead avoidance

| Rule | Enforcement |
|---|---|
| **Wednesday-to-Wednesday return** is the target. Signal computed at Wed close, return measured Wed close → next Wed close. | `build_target` does `weekly_close.pct_change().shift(-1)`. ✓ |
| **Fundamental publication lag.** SimFin "Report Date" is shifted forward by `pub_lag_days = 90` before being forward-filled to daily frequency. | `fetch_fundamentals_simfin` applies `lag = pd.DateOffset(days=90)`. ✓ |
| **Earnings-revision timestamps** must be strictly point-in-time (not the as-of date of a vendor snapshot). | Phase 2 — not yet implemented; document the requirement when sourcing IBES/FMP/Refinitiv. |
| **Execution lag.** Signal calculated at Wed close, but trades fill at Thu close (lag = 1d minimum). Returns must be measured from Thu close → next Thu close when reporting net-of-cost performance. Lag = 0 is leakage and must never headline. | `configs/default.yaml` sets `execution_lag_days: 1`. `src/main.py` shifts `close` by `-exec_lag` before resampling. `alpha_decay_analysis` sweeps lags 0-4 as a diagnostic curve only. Production paths must refuse `exec_lag < 1`. ✓ for current config; latent risk if `execution_lag_days` is ever set to 0 or omitted. |

## Survivorship bias

The Phase 1 universe comes from `pd.read_html(...List_of_S&P_500_companies)`. This is
**current** membership only — companies that were in the index in 2023 but were
delisted before 2026 are absent. Effect: backtest is slightly optimistic on the
short side and pessimistic on the long side (delistings tend to follow drawdowns).
Phase 2 must source point-in-time membership (e.g. via CRSP or vendor snapshot).

`src/data/universe.py:get_sp500_constituents` docstring already flags this.

## Neutralization ordering

The paper's recipe (p.13) is **fixed**: winsorize → iterative z-score (10×) →
cap at ±3 → subtract region × industry median → impute missing with 0. Skipping
or reordering changes what the model learns:

- **Industry-neutralize before z-scoring** → the variance removed by neutralization
  inflates the standardized scale, and stocks in low-dispersion industries get
  amplified scores.
- **No winsorization** → outliers dominate the z-score and the model learns the
  tails.
- **No iteration** → a single z-score pass is sensitive to remaining outliers; the
  10× iteration is what makes the standardization robust.

The codebase has `full_neutralization` that does the right ordering (modulo the
missing ±3 cap and missing-data rule). It is **not currently wired** into
`main.py`. Top-priority fix in [`007`](007-implementation-plan.md).

## Cost model

- **1.5 bps per side**, applied to 2-way turnover.
- **No market impact, no slippage** beyond the bps charge.
- **No borrow cost on shorts.** This is a known approximation; the paper's headline
  numbers also do not subtract borrow.

## Quintile peer group

- Phase 1: quintiles formed across the entire universe, equal-weight within Q1/Q5.
- Phase 2: quintiles formed within (region × GICS industry) buckets, equal-weight
  within each (peer-group, quintile) cell. This matches paper p.5.

## Validation checklist

These are the falsifiable predictions the build must reproduce. If the first three
fail, the model is not learning what the paper says it should learn — pause Phase 2
and investigate.

| Check | Expected result | Source |
|---|---|---|
| Top SHAP feature | `R1W` | paper p.7 |
| #2 SHAP feature | `UPDOWN1W` (Phase 2 only — needs earnings revisions) | paper p.7 |
| Alpha decay | Monotone decline 0d → 4d lag, ~50% of IR retained at 1d | paper p.7 |
| Weekday effect | Thursday signal > Monday signal | paper p.8 |
| Quintile spread | Monotonic Q1 → Q5 return | implicit in paper |
| ML vs R1W-only baseline | Higher net-of-cost IR | paper p.6 |
| ML vs earnings-filtered R1W baseline | Higher net-of-cost IR | paper p.6 |
| Turnover | ML lower than basic R1W reversal (paper: 212% vs 317% per week) | paper p.9 |

## Reference numbers (US, paper p.10)

After all costs and constraints:

| Metric | Paper value |
|---|---|
| Annualized return | 8.6% |
| Annualized vol | 8.5% |
| Information ratio | 1.0 |
| Max drawdown | 18.0% |

The Phase 1 backtest should land in this neighborhood (allowing for a much shorter
sample window, factor subset, and known gaps like the missing UPDOWN1W feature).
A Phase-1 IR materially below 0.5 or above 2.0 is a red flag that something in
the pipeline is wrong.
