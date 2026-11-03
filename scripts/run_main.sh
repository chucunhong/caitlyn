#!/usr/bin/env bash
# Main CAITLYN runs on the two backbones (sequential: the plan allows 3 concurrent calls in total).
# Usage: bash scripts/run_main.sh <python>
PY="$1"
cd "$(dirname "$0")/.."
mkdir -p results/_logs
for MODEL in default deepseek-v4.1-flash; do
  for DS in german credit_scoring adult; do
    LOG="results/_logs/main_${MODEL}_${DS}.log"
    echo "[$(date '+%F %T')] start $MODEL $DS" | tee -a results/_logs/main_progress.log
    if [ "$MODEL" = default ]; then
      "$PY" -u scripts/run_caitlyn.py --dataset "$DS" > "$LOG" 2>&1
    else
      "$PY" -u scripts/run_caitlyn.py --dataset "$DS" --model "$MODEL" > "$LOG" 2>&1
    fi
    echo "[$(date '+%F %T')] done  $MODEL $DS (exit $?)" | tee -a results/_logs/main_progress.log
  done
done
echo "[$(date '+%F %T')] ALL DONE" | tee -a results/_logs/main_progress.log
