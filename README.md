---
title: ML Short Reversion
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ML Short-Horizon Equity Reversion

Machine-learning strategy for weekly mean reversion in developed-market equities,
replicating and extending `ML_Mean_Reversion.pdf`. Predicts forward 1-week stock
returns from a cross-sectionally neutralized factor panel using an ensemble of
XGBoost, LightGBM, and RandomForest, with SHAP attribution per member.

> **Public build.** The code, the backtest, and the full SHAP attribution are
> here. The *live signal book* is not — current picks, the picks ledger and the
> per-name research are kept local. See "What is and isn't published" below.

## What the pages show

| Page | Contents |
|---|---|
| Overview | Headline IR / return / drawdown / turnover vs the R1W baseline |
| Predictions | Per-week cross-sectional prediction distributions |
| SHAP | Per-member feature attribution (beeswarm + bar) |
| Alpha decay | Annualized IR across 0-4 day execution lags |
| Weekday effect | IR by signal weekday (Mon-Fri) |
| Parameter explorer | Backtest sensitivity to config knobs |
| Report | Multi-agent research note (briefing → critique → synthesis) |
| Live forecast | Next-rebalance long/short picks — *empty in the public build* |
| Track record | Realized performance of past picks — *empty in the public build* |
| Deep dive | Per-name fundamentals / news / technicals — *empty in the public build* |

The last three read the live signal book, which is not published. Each page
renders an empty state naming the command that regenerates it locally.

## Current run

Signal day **THU**, 67 factors, 740 backtest weeks, data through **2026-08-14**.

All figures below are over the **same 740 weeks** (2012-06-21 → 2026-08-20), net
of 1.5 bps/side and a 1-day execution lag.

| Model | IR | Ann. return | Max drawdown |
|---|---|---|---|
| xgboost | 1.382 | 9.40% | −12.4% |
| lightgbm | 1.295 | 9.08% | −9.8% |
| random_forest | 1.192 | 8.07% | −8.1% |
| **ensemble** | **1.334** | 8.70% | **−7.8%** |
| R1W baseline | 0.596 | 6.93% | −25.2% |

`R1W` is the top SHAP feature (#1 of 67 in both boosters), consistent with the
paper's core hypothesis.

**Note on the baseline figure.** Earlier versions of this table quoted the R1W
baseline at **0.896**, which was not comparable: the baseline needs no training
and runs from 2006-01-12, while the model only starts after its 260+52-week
warmup at 2012-06-21. Scoring each over its own history gave the baseline 336
extra weeks the model never traded — including the GFC, when reversal did
unusually well. On matched weeks the baseline is **0.596**, so the model's edge
is larger than previously reported (+0.738 rather than +0.438), and the drawdown
gap (−7.8% vs −25.2%) was hidden entirely. The Overview page now clips all curves
to a common start and shows a like-for-like table.

**Caveat on these numbers:** they are not a clean read on any single change.
Relative to the previous run, the signal day moved WED→THU, the HMM regime block
was removed, the MLP member was dropped, and the data was extended — and xgboost
gained +0.67 despite no changes to it, which sets the noise floor. See
`.specify/007-implementation-plan.md`.

**One change has been isolated.** A controlled A/B of the LightGBM
random-hyperparameter bag (2026-08-17, same 740 weeks, only that flag changed)
put the LGB member at **1.295 with the bag vs 0.931 without** — the ensemble
moved just 1.334 → 1.286, since `rank_mean` dilutes a one-member gain across
three. xgboost and random_forest came out bit-identical in both arms, which is
what makes the comparison trustworthy. The bag is on by default; the ensemble-level
delta (+0.048) is inside the noise floor above, so the member lift is the result
worth quoting.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # full set, includes training deps
streamlit run app/streamlit_app.py
```

The Space itself installs `requirements-space.txt` — a viewer-only subset that
omits torch / shap / lightgbm / xgboost / modal, none of which the app imports.

## Deployment

The Space is a git remote; there is no CI in between:

```bash
git push origin master   # GitHub (public, source of truth)
./deploy_hf.sh           # Hugging Face Space (public, runtime)
```

Use `deploy_hf.sh` rather than pushing to `hf` directly: it builds a single-commit
orphan branch in a throwaway worktree with every binary forced through LFS, which
is what the Hub's pre-receive hook requires. It also means **the Space carries no
git history**, so nothing that was ever committed here is recoverable from it.

**No secrets are required.** The Space is a pure viewer over the committed
artifacts and makes no outbound API calls: the only fetching code path
(`return_calibration.build_calibration`) is never invoked by the app, and it
imports `fetch_price_data` inside the function, so nothing network-related is
even loaded. Refreshing data and retraining happen locally, not on the Space.

## What is and isn't published

Both the GitHub repo and the Space are **public**. The split is deliberate.

**Published** — everything needed to audit or reproduce the research: all source,
configs and tests; the 740-week backtest outputs (`portfolio_returns`,
`predictions`, `alpha_decay`, `weekday_effect`, calibration); full SHAP values and
features per ensemble member; and the multi-agent report prose.

**Not published** — the live signal book, kept out by `.gitignore` and absent from
git history:

| Path | Why |
|---|---|
| `data/processed/forecasts/` | current long/short picks — a tradeable signal |
| `data/processed/deepdive/` | per-name research over those picks |
| `data/processed/forecast_picks.db` | picks ledger |
| `data/processed/live_trackrecord.db` | realized performance of published picks |
| `data/processed/forecast_track_record_*.parquet` | per-signal-day track record |
| `data/processed/orders_*.csv`, `paper_positions_*.csv` | the live trading book |
| `data/processed/reports/**/research_pack.json` | embeds the source paper verbatim |

`predictions.parquet` ends at the last backtest week (2026-08-20). On a 1-week
holding horizon that ranking has long since expired, which is why it ships.

**The source paper is not redistributed.** `ML_Mean_Reversion.pdf` is third-party
research and has never been committed. The report pipeline reads it locally and
writes it into `research_pack.json`, which is why that file is excluded.

## Disclaimer

Research code, published for review. Nothing here is investment advice, and the
backtest is not a claim about future returns — it carries the usual caveats
(survivorship bias in the UK/CA snapshot universes, a cost model rather than
real fills, and the noise floor documented above).

## License

MIT — see [`LICENSE`](LICENSE).
