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

> **Private Space.** This app surfaces live long/short positions and the full
> factor model. It is deliberately not public — see "Visibility" below before
> changing that.

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
| Live forecast | Next-rebalance long/short picks per signal day (WED/THU/FRI) |
| Track record | Realized performance of past published picks |

## Current run

Signal day **THU**, 67 factors, 740 backtest weeks, data through **2026-08-14**.

| Model | IR |
|---|---|
| xgboost | 1.382 |
| lightgbm | 1.295 |
| random_forest | 1.192 |
| **ensemble** | **1.334** |
| R1W baseline | 0.896 |

`R1W` is the top SHAP feature (#1 of 67 in both boosters), consistent with the
paper's core hypothesis.

**Caveat on these numbers:** they are not a clean read on any single change.
Relative to the previous run, the signal day moved WED→THU, the HMM regime block
was removed, the MLP member was dropped, and the data was extended — and xgboost
gained +0.67 despite no changes to it, which sets the noise floor. See
`.specify/007-implementation-plan.md`.

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
git push origin macro-fmp-premium-fix   # GitHub (private, source of truth)
git push hf HEAD:main                   # Hugging Face Space (private, runtime)
```

`FMP_API_KEY` must be set in **Space Settings → Variables and secrets** for the
pages that refresh data; the committed artifacts render without it.

## Visibility

Both the GitHub repo and this Space are **private**. The app exposes current
positions, per-name SHAP rationale, and the live track record. Making it public
publishes the signal, and a public Space URL can be indexed and scraped even if
later deleted.

## License

MIT — see [`LICENSE`](LICENSE).
