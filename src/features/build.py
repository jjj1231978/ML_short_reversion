"""Orchestrate feature matrix construction from raw data."""

import pandas as pd

from src.config import PROCESSED_DIR, load_config
from src.features import factors


def resample_to_wednesday(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to Wednesday close values."""
    return df.resample("W-WED").last()


def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    market_close: pd.Series | None = None,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Build weekly feature matrix from daily price and fundamental data.

    Args:
        close: Daily close prices, columns = tickers.
        volume: Daily volume, columns = tickers.
        market_close: Market index close (e.g. SPY) for beta calculation.
        fundamentals: Dict from fetch_fundamentals_simfin with keys
            'PE', 'PB', 'ROE', 'GPOA' — each a daily DataFrame (dates x tickers).
        cfg: Config dict.

    Returns a MultiIndex DataFrame: (date, ticker) -> feature columns.
    """
    if cfg is None:
        cfg = load_config()

    feature_frames = {
        # Price reversals
        "R1W": factors.r1w(close).pipe(resample_to_wednesday),
        "RSI5D": factors.rsi(close, window=5).pipe(resample_to_wednesday),
        "RSI14": factors.rsi(close, window=14).pipe(resample_to_wednesday),
        # Price momentum
        "R3M1M": factors.momentum(close, lookback=63, skip=21).pipe(resample_to_wednesday),
        "R12M1M": factors.momentum(close, lookback=252, skip=21).pipe(resample_to_wednesday),
        # Low risk
        "VOL6M": factors.rolling_volatility(close, window=126).pipe(resample_to_wednesday),
    }

    if market_close is not None:
        feature_frames["BETA6M"] = factors.rolling_beta(
            close, market_close, window=126
        ).pipe(resample_to_wednesday)

    # Fundamental factors (daily-aligned from SimFin, resample to Wednesday)
    if fundamentals:
        for name in ["PE", "PB", "ROE", "GPOA"]:
            if name in fundamentals:
                feature_frames[name] = resample_to_wednesday(fundamentals[name])

    # Stack into (date, ticker) -> features
    combined = pd.concat(feature_frames, axis=1)
    # combined has columns = MultiIndex (feature_name, ticker)
    # Reshape to (date, ticker) rows x feature columns
    stacked = combined.stack(level=-1)
    stacked.index.names = ["date", "ticker"]

    return stacked


def build_target(close: pd.DataFrame) -> pd.DataFrame:
    """Forward 1-week return, Wednesday to Wednesday, cross-sectionally z-scored."""
    weekly_close = resample_to_wednesday(close)
    fwd_ret = weekly_close.pct_change().shift(-1)  # forward return

    # Cross-sectional z-score
    mean = fwd_ret.mean(axis=1)
    std = fwd_ret.std(axis=1).replace(0, float("nan"))
    zscored = fwd_ret.sub(mean, axis=0).div(std, axis=0)

    # Stack to (date, ticker)
    stacked = zscored.stack()
    stacked.name = "target"
    stacked.index.names = ["date", "ticker"]
    return stacked.to_frame()


if __name__ == "__main__":
    cfg = load_config()
    print("Feature build module — run via src.main or import directly.")
