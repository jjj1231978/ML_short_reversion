#!/usr/bin/env bash
# Launch the full THU-signal-day backtest detached, so it survives the terminal/session.
# ~16-18h (4-member ensemble: xgb+lgb+rf+mlp). Data is cached; config is already THU.
# Usage:  ./run_thu_backtest.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/thu_backtest_${TS}.log"
echo "######## FULL BACKTEST (signal_day=THU) START ${TS} ########" | tee "$LOG"
nohup .venv/bin/python -m src.main >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > logs/thu_backtest.pid
echo "launched PID $PID -> $LOG"
echo "monitor:  tail -f $LOG   |   grep -E 'Phase|saved' $LOG"
