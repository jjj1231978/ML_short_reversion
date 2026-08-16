"""Per-name research layer over the model's live picks.

`src.predict` says *which* names the model wants and *which factors* moved the
score. This package answers the next question — what is actually happening at
the company — along three angles:

  fundamentals : last reported earnings vs estimate, next earnings + consensus
                 expectation, TTM valuation multiples          (FMP)
  theme / news : what narrative is driving the stock right now, and the
                 headline risks in the last month     (FMP news US, Tavily all)
  technicals   : RSI / moving averages / 52w range, reduced to a deterministic
                 trend-or-reversal verdict                     (derived local)

The batch job (`python -m src.research deepdive`) writes one JSON dossier per
(signal_day, target_date) under `data/processed/deepdive/`; the Streamlit
"Deep dive" page renders it. The page never fetches unless explicitly unlocked
— see `DEEPDIVE_ALLOW_REFRESH` in `app/pages/10_deep_dive.py`.
"""
