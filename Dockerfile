# Hugging Face Space runtime (docker SDK, CPU Basic).
#
# Mirrors the pattern in ../DL-research-demo, with one difference: the Streamlit
# entrypoint here lives at app/streamlit_app.py rather than the repo root, and
# src/config.py resolves PROJECT_ROOT from __file__, so the whole repo must be
# copied to a single WORKDIR for data/processed to resolve.

FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker layer-caches them across content-only pushes.
# requirements-space.txt is the slim viewer set — see that file for why the
# training deps (torch/shap/lightgbm/xgboost/modal) are intentionally absent.
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# Copy the rest of the app (code + committed data/processed artifacts).
COPY . .

# HF Spaces route traffic to port 7860.
EXPOSE 7860

# Belt-and-suspenders: these also live in .streamlit/config.toml. The README
# front-matter app_port MUST agree with both or the Space renders blank.
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app

CMD ["streamlit", "run", "app/streamlit_app.py"]
