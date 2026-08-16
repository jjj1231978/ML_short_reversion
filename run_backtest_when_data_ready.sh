#!/usr/bin/env bash
# Wait for the in-flight FMP backfill to finish, verify the refreshed cache is
# actually usable, then launch the full THU backtest detached.
#
# Guarded on purpose: a 20h backtest started on a truncated cache (backfill
# killed, FMP 402/429 partway) would burn a day and produce results silently
# based on stale prices. If any check fails this refuses to launch and says why.
set -uo pipefail
cd "$(dirname "$0")"

DATA_LOG="logs/data_update_2026-08-15.log"
MIN_PRICE_DATE="2026-08-07"   # allow a few days' slack vs end_date 2026-08-14

echo "waiting for backfill to exit..."
while pgrep -f "src.data backfill" >/dev/null; do sleep 30; done
echo "backfill process exited at $(date)"

if ! grep -q "FMP backfill complete" "$DATA_LOG"; then
    echo "REFUSING TO LAUNCH: '$DATA_LOG' has no completion line — backfill did not finish cleanly."
    tail -5 "$DATA_LOG"
    exit 1
fi

# The completion line alone does not prove the rows landed: individual symbol
# fetches are caught and logged rather than raised. Check the cache directly.
.venv/bin/python - "$MIN_PRICE_DATE" <<'PYCHECK'
import sys, pandas as pd
from pathlib import Path
floor = pd.Timestamp(sys.argv[1])
d = Path.home() / "data_lake/fmp/prices/by_symbol"
maxes = []
for sym in ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]:
    p = d / f"{sym}.parquet"
    if p.exists():
        maxes.append((sym, pd.to_datetime(pd.read_parquet(p)["date"]).max()))
if not maxes:
    print("CHECK FAILED: no benchmark price parquets found"); sys.exit(1)
for sym, m in maxes:
    print(f"  {sym}: last price {m.date()}")
stale = [s for s, m in maxes if m < floor]
if stale:
    print(f"CHECK FAILED: {stale} still behind {floor.date()} — cache not refreshed")
    sys.exit(1)
print("price cache check passed")
PYCHECK
if [ $? -ne 0 ]; then
    echo "REFUSING TO LAUNCH: price cache did not refresh to $MIN_PRICE_DATE."
    exit 1
fi

# Grades top-up. The main backfill only fetches tickers MISSING from the grades
# cache; tickers already cached keep whatever the last full fetch saw, because
# FMP's /grades endpoint has no "from" cutoff. Without this pass UPDOWN1W_RATINGS
# is silently frozen at the previous refresh. Prices/fundamentals are skipped —
# they were just refreshed above.
GLOG="logs/grades_topup_2026-08-15.log"
echo "topping up analyst grades for already-cached tickers -> $GLOG"
.venv/bin/python -m src.data backfill --regions US,UK,CA --topup-grades \
    --skip-prices --skip-benchmarks --skip-fundamentals > "$GLOG" 2>&1
if ! grep -q "FMP backfill complete" "$GLOG"; then
    echo "REFUSING TO LAUNCH: grades top-up did not complete cleanly."
    tail -5 "$GLOG"
    exit 1
fi
echo "grades top-up done at $(date)"

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/thu_backtest_${TS}.log"
echo "######## FULL BACKTEST (signal_day=THU, LGB random ensemble via Modal) START ${TS} ########" | tee "$LOG"
nohup .venv/bin/python -m src.main >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > logs/thu_backtest.pid
echo "launched PID $PID -> $LOG"
