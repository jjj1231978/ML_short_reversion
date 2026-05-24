"""Report viewer: renders the multi-agent reporting workflow output."""

from __future__ import annotations

import difflib
import json

import streamlit as st

from app.lib import data_loader as dl

st.title("Report")

runs = dl.list_report_runs()
if not runs:
    st.warning(
        "No report runs found in `data/processed/reports/`. "
        "Generate one with: `python -m src.reporting.run_report`"
    )
    st.stop()

with st.sidebar:
    st.header("Run selector")
    sel = st.selectbox("Report run", runs, index=0)

run = dl.load_report_run(sel)

tabs = st.tabs(["Final", "Draft", "Critique", "Meta", "Diff (draft → final)"])

with tabs[0]:
    if run.get("final"):
        st.markdown(run["final"])
    else:
        st.info("`final.md` not present in this run.")

with tabs[1]:
    if run.get("draft"):
        st.markdown(run["draft"])
    else:
        st.info("`draft.md` not present in this run.")

with tabs[2]:
    if run.get("critique"):
        st.markdown(run["critique"])
    else:
        st.info("`critique.md` not present in this run.")

with tabs[3]:
    meta = run.get("meta")
    if meta:
        st.json(meta)
    else:
        st.info("`meta.json` not present in this run.")

with tabs[4]:
    if run.get("draft") and run.get("final"):
        diff = difflib.unified_diff(
            run["draft"].splitlines(),
            run["final"].splitlines(),
            fromfile="draft.md",
            tofile="final.md",
            lineterm="",
            n=3,
        )
        st.code("\n".join(diff), language="diff")
    else:
        st.info("Need both draft.md and final.md to render a diff.")
