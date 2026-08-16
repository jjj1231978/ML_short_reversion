"""Size the latest live forecast into a concrete, reviewable order list.

Adapted from ../ML_short_sentiment/scripts/ibkr_meta_orders.py. That fork sizes a
meta-strategy net book (`meta_forecasts/`); this project has no meta layer, so we
size directly from the per-signal-day forecast parquet written by src/predict.py.

    python -m scripts.build_orders --signal-day THU --capital 100000 --top 15

This writes a CSV and prints the book. It PLACES NOTHING — order transmission is a
separate, deliberate step. Pair it with --liquidate to also emit closing orders for
everything currently held.

Sizing: prediction ranks are converted to equal-weight long/short legs, normalized
to gross 1.0, so --capital is the total GROSS dollars deployed (roughly half long,
half short, dollar-neutral). Prices are the latest cached close in each name's local
currency (US = USD, CA .TO = CAD); est_usd converts CAD at --cadusd.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Stale FMP ticker -> current IBKR ticker. The cached FMP universe can lag corporate
# ticker changes, so renamed names are mapped here to route to the right contract.
SYMBOL_OVERRIDE = {"SATS": "ECHO"}


def _latest_forecast(signal_day: str) -> tuple[pd.DataFrame, str]:
    d = PROCESSED_DIR / "forecasts" / signal_day.upper()
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        raise SystemExit(f"No forecast under {d}; run `python -m src.predict --signal-day {signal_day}` first.")
    return pd.read_parquet(files[-1]), files[-1].stem


def _to_ibkr(ticker: str) -> tuple[str, str, str]:
    """(ibkr_symbol, exchange, currency) for a cached FMP ticker."""
    t = SYMBOL_OVERRIDE.get(ticker, ticker)
    if t.endswith(".TO"):
        return t[:-3].replace("-", " "), "TSE", "CAD"
    if t.endswith(".L"):
        return t[:-2], "LSE", "GBP"
    return t, "SMART", "USD"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-day", default="THU")
    ap.add_argument("--capital", type=float, required=True,
                    help="Total GROSS dollars to deploy (~half long / half short).")
    ap.add_argument("--top", type=int, default=15, help="Names per side.")
    # US-only by default: IBKR rejects Canadian products over the API on this
    # account ("Error 201: API/CTCI orders for Canadian products are not allowed"),
    # even though the account IS permissioned to trade Canada manually. Placing a
    # US+CA book therefore silently drops the CA leg — and since that leg is
    # long-skewed, what survives is badly net-short rather than dollar-neutral.
    # Pass --regions US,CA explicitly if the API restriction is ever lifted.
    ap.add_argument("--regions", default="US", help="Comma-separated regions to trade.")
    ap.add_argument("--cadusd", type=float, default=0.73, help="CAD->USD rate for est_usd.")
    ap.add_argument("--liquidate", type=Path, default=None,
                    help="CSV of current positions (ticker,position) to emit closing orders for.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sd = args.signal_day.upper()
    cfg = load_config()
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]

    fc, target = _latest_forecast(sd)
    book = fc[fc["eligible"]] if "eligible" in fc.columns else fc.copy()
    book = book[book["region"].isin(regions)]
    if book.empty:
        raise SystemExit(f"No eligible names in regions {regions} for target {target}.")

    ranked = book.sort_values("prediction", ascending=False)
    longs, shorts = ranked.head(args.top), ranked.tail(args.top)
    if len(longs) < args.top or len(shorts) < args.top:
        log.warning(f"Only {len(longs)} long / {len(shorts)} short names available (asked {args.top}).")

    # Equal weight within each leg, normalized so |weights| sum to 1.0 (gross).
    n = len(longs) + len(shorts)
    sel = pd.concat([longs, shorts])
    sel = sel.assign(side=["BUY"] * len(longs) + ["SELL"] * len(shorts))
    sel["weight"] = [1.0 / n] * len(longs) + [-1.0 / n] * len(shorts)

    from src.data.fetch import fetch_price_data
    tickers = [t[1] if isinstance(t, tuple) else t for t in sel.index]
    close = fetch_price_data(
        tickers, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"],
        source=cfg["data"].get("price_source", "fmp"), skip_fetch=True,
    )["Close"]
    last_px = close.ffill().iloc[-1]

    rows = []
    for (idx, r), w in zip(sel.iterrows(), sel["weight"]):
        tk = idx[1] if isinstance(idx, tuple) else idx
        sym, exch, ccy = _to_ibkr(str(tk))
        px = last_px.get(tk)
        if px is None or not (px > 0):
            log.warning(f"No cached price for {tk}; dropped.")
            continue
        # Local-currency notional, so share counts are right on the TSE too.
        usd_target = abs(w) * args.capital
        local_target = usd_target / (args.cadusd if ccy == "CAD" else 1.0)
        sh = round(local_target / px)
        if sh <= 0:
            continue
        rows.append({
            "ticker": tk, "ibkr_symbol": sym, "exchange": exch, "currency": ccy,
            "action": "BUY" if w > 0 else "SELL", "shares": int(sh),
            "price": round(float(px), 4),
            "est_usd": round(sh * px * (args.cadusd if ccy == "CAD" else 1.0), 2),
            "region": r["region"], "prediction": round(float(r["prediction"]), 4),
            "order_type": "TARGET",
        })

    orders = pd.DataFrame(rows)

    liq = pd.DataFrame()
    if args.liquidate and args.liquidate.exists():
        pos = pd.read_csv(args.liquidate)
        lrows = []
        for _, p in pos.iterrows():
            q = float(p["position"])
            if q == 0:
                continue
            lrows.append({
                "ticker": p["ticker"], "ibkr_symbol": p.get("ibkr_symbol", p["ticker"]),
                "exchange": p.get("exchange", "SMART"), "currency": p.get("currency", "USD"),
                # Closing order is the opposite sign of the held quantity.
                "action": "SELL" if q > 0 else "BUY", "shares": int(abs(q)),
                "price": p.get("market_price", float("nan")),
                "est_usd": round(abs(float(p.get("market_value", 0))), 2),
                "region": p.get("region", "?"), "prediction": float("nan"),
                "order_type": "LIQUIDATE",
            })
        liq = pd.DataFrame(lrows)

    out = args.out or PROCESSED_DIR / f"orders_{sd}_{target}.csv"
    allo = pd.concat([liq, orders], ignore_index=True) if not liq.empty else orders
    allo.to_csv(out, index=False)

    gross = orders["est_usd"].sum()
    lo = orders[orders.action == "BUY"]["est_usd"].sum()
    sh_ = orders[orders.action == "SELL"]["est_usd"].sum()
    log.info(f"Target {target}: {len(orders)} new orders, {len(liq)} liquidations")
    log.info(f"Gross ${gross:,.0f} (long ${lo:,.0f} / short ${sh_:,.0f}, net ${lo - sh_:+,.0f})")
    if not liq.empty:
        log.info(f"Liquidating ${liq['est_usd'].sum():,.0f} of existing positions")
    log.info(f"Wrote {out}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        log.info("\n" + allo.to_string(index=False))


if __name__ == "__main__":
    main()
