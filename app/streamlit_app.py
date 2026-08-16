"""Streamlit entry point for the ML short-reversion results viewer.

Run from the repo root:

    streamlit run app/streamlit_app.py

Multi-page navigation via st.navigation. Pages are defined in app/pages/.
The viewer is read-only — it loads cached parquets from data/processed/ and
re-derives parameter-explorer metrics on demand. It never triggers data
fetches or model retraining.

One deliberate exception: the Deep dive page offers a single-name refresh
against FMP and the news search, and it is off unless DEEPDIVE_ALLOW_REFRESH=1
is set *and* the API keys are present. Unset, that page is read-only like the
rest and renders only its cached JSON artifact.
"""

import sys
from pathlib import Path

# Allow `from src.*` and `from app.*` imports when running via streamlit.
#
# NOTE: this alone is not sufficient. Streamlit puts the *entrypoint's* directory
# (app/) on sys.path[0] and restores sys.path around each script run, so this
# insert does not survive into the page modules that st.navigation executes —
# they fail with "No module named 'app'" / "'src'". The repo root must therefore
# be on PYTHONPATH when launching:
#
#     PYTHONPATH=. streamlit run app/streamlit_app.py
#
# The Dockerfile already sets PYTHONPATH=/app, so the Hugging Face Space is fine.
# Remove-then-insert rather than an `if not in sys.path` guard: it is idempotent
# across reruns and always leaves the root at the front.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="ML Short Reversion",
    page_icon=":bar_chart:",
    layout="wide",
)

PAGES_DIR = Path(__file__).resolve().parent / "pages"

pages = [
    st.Page(PAGES_DIR / "01_overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page(PAGES_DIR / "02_predictions.py", title="Predictions", icon=":material/leaderboard:"),
    st.Page(PAGES_DIR / "03_shap.py", title="SHAP", icon=":material/insights:"),
    st.Page(PAGES_DIR / "04_alpha_decay.py", title="Alpha decay", icon=":material/timeline:"),
    st.Page(PAGES_DIR / "05_weekday_effect.py", title="Weekday effect", icon=":material/calendar_today:"),
    st.Page(PAGES_DIR / "06_parameter_explorer.py", title="Parameter explorer", icon=":material/tune:"),
    st.Page(PAGES_DIR / "07_report.py", title="Report", icon=":material/description:"),
    st.Page(PAGES_DIR / "08_live_forecast.py", title="Live forecast", icon=":material/online_prediction:"),
    st.Page(PAGES_DIR / "09_track_record.py", title="Track record", icon=":material/scoreboard:"),
    st.Page(PAGES_DIR / "10_deep_dive.py", title="Deep dive", icon=":material/travel_explore:"),
]

nav = st.navigation(pages)
nav.run()
