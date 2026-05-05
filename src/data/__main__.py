"""Allow `python -m src.data` to run data fetching."""
from src.data.fetch import fetch_price_data
from src.data.universe import get_sp500_tickers
from src.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    tickers = get_sp500_tickers()
    source = cfg["data"].get("price_source", "databento")
    print(f"Fetching data for {len(tickers)} tickers via {source}...")
    prices = fetch_price_data(
        tickers,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        source=source,
    )
    print(f"Price data shape: {prices['Close'].shape}")
