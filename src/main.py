"""Main pipeline orchestrator: data -> features -> model -> backtest -> diagnostics."""

import logging

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, load_config
from src.data.fetch import fetch_price_data, fetch_fundamentals_simfin
from src.data.universe import get_sp500_constituents, filter_by_gics_sector, get_gics_sector_map
from src.features.build import build_feature_matrix, build_target
from src.features.neutralize import full_neutralization
from src.model.train import rolling_train_predict
from src.backtest.portfolio import (
    build_long_short_portfolio,
    compute_performance_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(config_name: str = "default") -> dict:
    """Execute the full pipeline end-to-end."""
    cfg = load_config(config_name)

    # --- Phase 1: Data Acquisition ---
    log.info("Phase 1: Fetching universe...")
    constituents = get_sp500_constituents()
    sector_map = dict(zip(constituents["Symbol"], constituents["GICS Sector"]))
    tickers = constituents["Symbol"].tolist()
    log.info(f"Raw universe: {len(tickers)} tickers")

    # Exclude financials (GICS sector 40 = "Financials")
    tickers = filter_by_gics_sector(tickers, sector_map, exclude_sectors=["Financials"])
    log.info(f"After excluding Financials: {len(tickers)} tickers")

    log.info("Phase 1a: Fetching price data...")
    price_data = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        dataset=cfg["data"]["databento_dataset"],
    )
    close = price_data["Close"]
    volume = price_data["Volume"]
    log.info(f"Price data: {close.shape[0]} days x {close.shape[1]} tickers")

    log.info("Phase 1b: Fetching fundamentals...")
    fundamentals = fetch_fundamentals_simfin(
        tickers=tickers,
        pub_lag_days=cfg["data"]["fundamental_pub_lag_days"],
    )

    # Align fundamental columns to price columns (handle ticker format differences)
    common_tickers = close.columns.intersection(
        fundamentals.get("PE", pd.DataFrame()).columns
    )
    if len(common_tickers) > 0:
        log.info(f"Fundamental/price ticker overlap: {len(common_tickers)}")
    else:
        log.warning("No ticker overlap between price and fundamental data — check ticker formats")

    # --- Phase 2: Feature Engineering ---
    log.info("Phase 2: Building features...")
    features = build_feature_matrix(close, volume, fundamentals=fundamentals, cfg=cfg)
    target = build_target(close)
    log.info(f"Feature matrix: {features.shape}")

    # Neutralize features (per-factor, across stocks each week)
    # TODO: Wire in industry_map from sector_map for full neutralization
    log.info("Phase 2b: Neutralizing features...")

    # --- Phase 3: Model Training ---
    log.info("Phase 3: Training model...")
    predictions = rolling_train_predict(features, target, cfg=cfg)
    log.info(f"Predictions: {len(predictions)} observations")

    # --- Phase 4: Backtesting ---
    log.info("Phase 4: Running backtest...")
    from src.features.build import resample_to_wednesday

    weekly_returns = resample_to_wednesday(close).pct_change().shift(-1)
    portfolio = build_long_short_portfolio(predictions, weekly_returns, cfg=cfg)
    metrics = compute_performance_metrics(portfolio)

    log.info("=== Backtest Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info(f"  {k}: {v:.4f}")
        else:
            log.info(f"  {k}: {v}")

    # --- Phase 5: Save Results ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(PROCESSED_DIR / "portfolio_returns.parquet")
    predictions.to_frame().to_parquet(PROCESSED_DIR / "predictions.parquet")
    log.info(f"Results saved to {PROCESSED_DIR}")

    return metrics


if __name__ == "__main__":
    run_pipeline()
