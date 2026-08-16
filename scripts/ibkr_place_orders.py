"""Place the forecast book into an IBKR PAPER account (reconcile-to-target).

Ported from ../ML_short_sentiment/scripts/ibkr_place_orders.py; reads the order CSV
written by `scripts.build_orders` rather than meta_orders_*.csv.

US-ONLY BY DEFAULT. The account is permissioned to trade Canada, but IBKR refuses
Canadian products over the API ("Error 201: API/CTCI orders for Canadian products
are not allowed"), so a US+CA book silently loses its CA leg on submission. Because
that leg is long-skewed, what survives is net-short rather than dollar-neutral —
which is why the region filter is a hard default rather than a preference.

ALL ORDERS ROUTE SMART. Direct-routed API orders are DISCARDED by IB's default
precautionary setting (error 10311 -> 201, order discarded) — this is silent, and a
dry-run cannot detect it because nothing is submitted. Contracts built from
ib.positions() carry the listing exchange (NYSE/NASDAQ), so closes must be rebuilt
onto SMART with primaryExchange rather than reused as-is.

SAFETY
  * Dry-run by DEFAULT — prints the plan and transmits nothing. Add --transmit to send.
  * Hard guard: refuses any account whose id does not start with 'DU' (paper).
  * Contracts are qualified up front; unresolvable symbols are dropped from the plan
    rather than silently cancelled mid-batch.
  * Market orders by default; --limit for a marketable limit, --passive-pct to rest.

RECONCILE, NOT LIQUIDATE-AND-REBUY
  The plan is the DELTA between the target book and current positions, so names held
  but not in the target are closed, and names in both are merely resized. That reaches
  the same end state as "liquidate everything then buy the book" while trading far
  less: an overlapping name would otherwise be round-tripped for no reason, paying
  spread twice. Use --flatten-first only if you specifically want every existing
  position closed as its own order.

    # 1) size the book (writes data/processed/orders_{DAY}_{target}.csv)
    python -m scripts.build_orders --signal-day THU --capital 100000 --top 15
    # 2) preview the IBKR rebalance (nothing sent)
    python -m scripts.ibkr_place_orders --signal-day THU
    # 3) submit to paper
    python -m scripts.ibkr_place_orders --signal-day THU --transmit
"""

from __future__ import annotations

import argparse
import logging
import subprocess

import pandas as pd

from src.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Stale FMP ticker -> current IBKR ticker. The cached FMP universe can lag corporate
# ticker changes, so renamed names are mapped here to route to the right contract.
SYMBOL_OVERRIDE = {"SATS": "ECHO"}


def _default_host() -> str:
    """WSL2 NAT: IB Gateway runs on Windows, reachable at the default-gateway IP.
    (Under WSL mirrored networking, pass --host 127.0.0.1.)"""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        return out.split()[2]
    except Exception:
        return "127.0.0.1"


def _latest_order_csv(signal_day: str):
    files = sorted(PROCESSED_DIR.glob(f"orders_{signal_day}_*.csv"))
    if not files:
        raise SystemExit(
            f"No orders_{signal_day}_*.csv in {PROCESSED_DIR} — "
            f"run `python -m scripts.build_orders --signal-day {signal_day} --capital N` first."
        )
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-day", default="THU")
    ap.add_argument("--host", default=None, help="Gateway host (default: auto-detect WSL host).")
    ap.add_argument("--port", type=int, default=4002, help="4002=Gateway paper, 7497=TWS paper.")
    ap.add_argument("--client-id", type=int, default=11)
    ap.add_argument("--transmit", action="store_true", help="Actually send orders (default: dry-run).")
    ap.add_argument("--limit", action="store_true", help="Marketable LIMIT instead of MARKET.")
    ap.add_argument("--band-bps", type=float, default=50.0, help="Marketable-limit band off last close (bps).")
    ap.add_argument("--passive-pct", type=float, default=None,
                    help="PASSIVE limit this %% away from close (BUY below, SELL above) — resting orders.")
    ap.add_argument("--tif", default="DAY", help="Time-in-force: DAY or GTC (use GTC for resting orders).")
    ap.add_argument("--good-after-time", default=None,
                    help="IB Good-After-Time: submit now, activate later. Held server-side, so the "
                         "gateway need not stay connected. Format 'YYYYMMDD HH:MM:SS TZ', "
                         "e.g. '20260817 09:30:00 US/Eastern'.")
    ap.add_argument("--flatten-first", action="store_true",
                    help="Close EVERY existing position as its own order instead of netting the "
                         "delta. Trades more and pays spread twice on overlapping names.")
    # See scripts/build_orders.py for why this defaults to US-only.
    ap.add_argument("--regions", default="US", help="Regions to trade from the order CSV.")
    args = ap.parse_args()
    sd = args.signal_day.upper()
    host = args.host or _default_host()

    from ib_async import IB, LimitOrder, MarketOrder, Stock, util
    util.logToConsole("ERROR")

    csv = _latest_order_csv(sd)
    log.info(f"Reading {csv.name}")
    allo = pd.read_csv(csv)
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    # Only TARGET rows define the book. LIQUIDATE rows in the CSV are informational —
    # the reconcile below closes held-but-untargeted names on its own.
    tgt_rows = allo[allo.get("order_type", "TARGET") == "TARGET"]
    tgt_rows = tgt_rows[tgt_rows["region"].isin(regions)]
    if tgt_rows.empty:
        raise SystemExit(f"No TARGET rows for regions {regions} in {csv.name}.")

    # Signed target shares per ticker (BUY = +, SELL = -), plus contract metadata.
    target = {r.ticker: (r.shares if r.action == "BUY" else -r.shares) for r in tgt_rows.itertuples()}
    meta = {r.ticker: (r.ibkr_symbol, r.exchange, r.currency) for r in tgt_rows.itertuples()}
    last = {r.ticker: r.price for r in tgt_rows.itertuples()}

    ib = IB()
    log.info(f"Connecting to {host}:{args.port} (clientId={args.client_id})...")
    ib.connect(host, args.port, clientId=args.client_id, timeout=20)
    try:
        acct = ib.managedAccounts()[0]
        if not acct.startswith("DU"):
            raise SystemExit(f"REFUSING: account {acct} is not a paper account (must start with 'DU').")
        log.info(f"Paper account: {acct}")

        # Fold renames onto the current IBKR symbol BEFORE the diff, so a renamed name
        # nets against the held share count instead of emitting a spurious
        # close-new + open-old pair.
        target = {SYMBOL_OVERRIDE.get(t, t): v for t, v in target.items()}
        meta = {SYMBOL_OVERRIDE.get(t, t): v for t, v in meta.items()}
        last = {SYMBOL_OVERRIDE.get(t, t): v for t, v in last.items()}

        # Key held positions by IBKR symbol; remember each one's real contract so a
        # close order routes to the right exchange/currency.
        cur, cur_contract = {}, {}
        for p in ib.positions():
            if p.account != acct:
                continue
            sym = p.contract.symbol
            cur[sym] = cur.get(sym, 0.0) + p.position
            cur_contract[sym] = p.contract

        if args.flatten_first:
            # Explicit close of everything, then the full target as fresh orders.
            plan_rows = []
            for sym, held in cur.items():
                if abs(held) >= 1:
                    plan_rows.append({"ticker": sym, "action": "SELL" if held > 0 else "BUY",
                                      "qty": int(abs(held)), "target": 0, "held": int(held),
                                      "ref_px": float("nan"), "lmt": None})
            for t, tgt in target.items():
                if abs(tgt) >= 1:
                    plan_rows.append({"ticker": t, "action": "BUY" if tgt > 0 else "SELL",
                                      "qty": int(abs(tgt)), "target": int(tgt), "held": 0,
                                      "ref_px": last.get(t, float("nan")), "lmt": None})
            plan = pd.DataFrame(plan_rows)
        else:
            tickers = sorted(set(target) | set(cur))
            plan_rows = []
            for t in tickers:
                tgt, held = target.get(t, 0.0), cur.get(t, 0.0)
                delta = round(tgt - held)
                if abs(delta) < 1:
                    continue
                plan_rows.append({"ticker": t, "action": "BUY" if delta > 0 else "SELL",
                                  "qty": int(abs(delta)), "target": int(tgt), "held": int(held),
                                  "ref_px": last.get(t, float("nan")), "lmt": None})
            plan = pd.DataFrame(plan_rows)

        if plan.empty:
            log.info("Already at target — no orders needed.")
            return

        # Limit pricing (needs a reference close; names being closed may not have one).
        def _price(row):
            ref = row["ref_px"]
            if pd.isna(ref):
                return None
            if args.passive_pct is not None:      # passive: buy below close, sell above
                s = -1 if row["action"] == "BUY" else 1
                return round(ref * (1 + s * args.passive_pct / 100), 2)
            if args.limit:                        # marketable: buy above, sell below
                s = 1 if row["action"] == "BUY" else -1
                return round(ref * (1 + s * args.band_bps / 1e4), 2)
            return None
        plan["lmt"] = plan.apply(_price, axis=1)

        # Resolve contracts up front so a bad symbol cannot be silently cancelled
        # mid-batch. Targeted names use the CSV's exchange/currency; names only being
        # closed reuse the contract IBKR already reports for the position.
        # ALWAYS route SMART, never to a named exchange. IB's default precautionary
        # setting ("Direct routed orders may result in higher trade fees", error
        # 10311) DISCARDS direct-routed API orders outright — error 201, order
        # discarded, silently. Two ways that bit us:
        #   - closes reused the contract from ib.positions(), which carries the
        #     listing exchange (NYSE/NASDAQ) rather than SMART;
        #   - the CA legs were built with exchange="TSE".
        # primaryExchange still disambiguates the listing without direct-routing.
        def _smart(sym: str, ccy: str, primary: str | None) -> Stock:
            c = Stock(str(sym), "SMART", str(ccy))
            if primary and primary.upper() not in ("SMART", "", "NAN"):
                c.primaryExchange = str(primary).upper()
            return c

        conmap = {}
        for t in plan["ticker"]:
            if t in meta:
                sym, exch, ccy = meta[t]
                conmap[t] = _smart(sym, ccy, exch)
            elif t in cur_contract:
                held_c = cur_contract[t]
                conmap[t] = _smart(held_c.symbol, held_c.currency,
                                   held_c.primaryExchange or held_c.exchange)
            else:
                conmap[t] = Stock(t, "SMART", "USD")
        qualified = ib.qualifyContracts(*conmap.values())
        ok = {(c.symbol, c.currency) for c in qualified}
        resolved = {t: c for t, c in conmap.items() if (c.symbol, c.currency) in ok}
        skipped = [t for t in conmap if t not in resolved]
        if skipped:
            log.warning(f"Skipping {len(skipped)} unresolved symbol(s) — no IBKR contract: {skipped}")
            plan = plan[~plan["ticker"].isin(skipped)].reset_index(drop=True)
        if plan.empty:
            log.info("No resolvable orders.")
            return

        plan["ccy"] = [resolved[t].currency for t in plan["ticker"]]
        gross = (plan["qty"] * plan["ref_px"]).sum(skipna=True)
        log.info(f"Rebalance plan: {len(plan)} orders "
                 f"({(plan.action == 'BUY').sum()} BUY / {(plan.action == 'SELL').sum()} SELL), "
                 f"~{gross:,.0f} traded (mixed ccy)")
        closing = plan[plan["target"] == 0]
        if len(closing):
            log.info(f"  of which {len(closing)} are closing positions no longer in the book")
        with pd.option_context("display.max_rows", None, "display.width", 150):
            log.info("\n" + plan.to_string(index=False))

        if not args.transmit:
            log.info("DRY-RUN — no orders sent. Re-run with --transmit to submit to paper.")
            return

        log.info(f"TRANSMITTING {len(plan)} orders to paper account {acct}...")
        trades = []
        for row in plan.itertuples():
            c = resolved[row.ticker]
            o = (LimitOrder(row.action, row.qty, row.lmt)
                 if row.lmt is not None and not pd.isna(row.lmt)
                 else MarketOrder(row.action, row.qty))
            o.tif = args.tif
            if args.good_after_time:
                o.goodAfterTime = args.good_after_time
            trades.append((row.ticker, ib.placeOrder(c, o)))
        ib.sleep(3)
        for tk, tr in trades:
            log.info(f"  {tk:10} {tr.order.action} {tr.order.totalQuantity} → {tr.orderStatus.status}")
        log.info(f"Submitted {len(trades)} orders to {acct}.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
