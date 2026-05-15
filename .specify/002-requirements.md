# 002 — Requirements

Specs for each component, broken down into "must match paper" vs "Phase 1 simplification".
The implementation plan in [`007`](007-implementation-plan.md) cross-references these.

## 1. Universe

| Requirement | Spec | Source |
|---|---|---|
| Region | Phase 1: US (S&P 500). Phase 2: + Europe + Japan large/mid cap | paper p.4 |
| Sector exclusions | GICS sector 40 (Financials) | paper p.4 footnote 2 |
| Liquidity floor | Drop stocks with 6-month rolling ADV < $3M | paper p.10 |
| Membership | Phase 1: current Wikipedia list. Phase 2: point-in-time | paper assumes PIT |
| Re-evaluation cadence | Weekly on Wednesday | paper p.4 |

## 2. Data sources

| Domain | Source | Cache location |
|---|---|---|
| Daily OHLCV | Databento `EQUS.MINI` schema `ohlcv-1d` | `~/data_lake/databento/equities/L0/ohlcv-1d/` |
| Fundamentals (PE, PB, ROE, GPOA) | SimFin TTM income / balance / prices | `~/data_lake/simfin/` |
| Earnings revisions (Phase 2) | TBD — IBES, FMP, or Refinitiv | TBD |
| Index constituents | Wikipedia (Phase 1); PIT vendor (Phase 2) | live HTTP, not cached |
| GICS sector / industry | Wikipedia (Phase 1); vendor (Phase 2) | live HTTP, not cached |

Raw data lives outside the repo (in `~/data_lake/`). Processed artifacts live in
`./data/processed/`. See [`005-outputs.md`](005-outputs.md).

## 3. Factor library

### Phase 1 — implemented set

| Factor | Group | Definition | Z-score group (paper) |
|---|---|---|---|
| `R1W` | Reversals | 5-day total return | Industry |
| `IREV1W` | Reversals | 1-week beta-adjusted residual return | Industry |
| `RSI5D` | Reversals | 5-day RSI | Industry |
| `RSI14` | Reversals | 14-day RSI | Industry |
| `R3M1M` | Momentum | 3-month return lagged 1 month | Universe |
| `R12M1M` | Momentum | 12-month return lagged 1 month | Universe |
| `VOL6M` | Low risk | 6-month annualized vol | Universe |
| `BETA6M` | Low risk | 6-month rolling beta vs SPX | Universe |
| `PE` | Value | Trailing P/E | Industry |
| `PB` | Value | Trailing P/B | Industry |
| `ROE` | Profitability | Trailing return on equity | Industry |
| `GPOA` | Profitability | Trailing gross profit / assets | Industry |

### Phase 2 — earnings momentum (must add)

The paper's #2 SHAP feature is `UPDOWN1W`. Without it the validation checklist
(top-2 features) cannot be tested.

| Factor | Group | Definition |
|---|---|---|
| `UPDOWN1W` | Earnings | 7-day rolling EPS up-revisions − down-revisions / total revisions |
| `UPDOWN3` | Earnings | 75-day version of above |
| `UPDOWN6` | Earnings | 180-day version of above |
| `SUE1W` | Earnings | (Current FY1 EPS − FY1 EPS 1w ago) / vol of EPS estimates |
| `SUE3` | Earnings | 3-month version of above |
| `SUE6` | Earnings | 6-month version of above |

### Phase 2 — additional factor groups (paper has 86 total)

Leverage (DE, NDE, EBITCOVER, …), Growth (5y trends and t-stats), Earnings Quality
(accruals, AQI, GMI, …), Capital Allocation (CAPEXPPE, STKP, EXTFA), more
Reversals (R1M, IREV1M, IREVVOL1W/1M, RSI30), more Momentum (residual variants).
Full list in paper Addendum I (pp. 13–14).

## 4. Target variable

- **Definition:** forward 1-week return, Wednesday close to Wednesday close.
- **Cross-sectional transform:** z-score across the universe each week (mean 0, std 1).
- **Source:** computed from the same OHLCV series as the features.
- **Codebase:** `src/features/build.py::build_target`.

## 5. Model

| Spec | Phase 1 value | Paper |
|---|---|---|
| Algorithm | XGBoost regressor (`reg:squarederror`) | "machine learning" — paper does not name the model class but the methodology and SHAP usage are consistent with gradient boosted trees |
| Tree depth | 4 | not stated |
| Learning rate | 0.03 | not stated |
| `n_estimators` | 2000 (early-stop on val) | not stated |
| `subsample` / `colsample_bytree` | 0.7 / 0.7 | not stated |
| Train window | 78 weeks (~1.5 yrs) | 520 weeks (~10 yrs) |
| Validation window | 26 weeks (~6 mo) | 104 weeks (~2 yrs) |
| Test window | 1 week | 1 week |
| Retrain cadence | Every 12 weeks | Every 12 weeks |
| Early stopping | 50 rounds | not stated |
| Feature inputs | neutralized factor matrix | neutralized factor matrix |

## 6. Backtest

| Spec | Value | Source |
|---|---|---|
| Rebalance day | Wednesday | paper p.4 |
| Long basket | Top quintile (Q5) | paper p.5 |
| Short basket | Bottom quintile (Q1) | paper p.5 |
| Quintile peer group | Phase 1: universe-wide. Phase 2: region × industry | paper p.5 |
| Weighting | Equal-weight within quintile | paper p.10 |
| ADV scaling | Stocks with ADV ≥ $20M get full equal weight; ADV < $20M scaled down proportionally | paper p.10 |
| Trading cost | 1.5 bps per side, applied to 2-way turnover | paper p.10 |
| Execution lag | 1 day (signal calculated Wed close, execute Thu close) | paper p.10 |

## 7. Diagnostics

Required outputs (see [`005-outputs.md`](005-outputs.md) for artifact paths):

- SHAP beeswarm and mean-|SHAP| bar chart of feature importance.
- Alpha decay curve: IR / annualized return as a function of execution lag (0–4 days).
- Weekday effect: same backtest run with signals computed on each weekday (Mon–Fri).
- Quintile spread: monotonic Q1 → Q5 returns.
- Cumulative L/S vs reversal-baseline-only vs earnings-filtered-baseline.
- Crisis-period table (paper p.11) — once Phase 2 has the history.
- Performance by market-return quintile (paper p.12) — once Phase 2 has the history.
