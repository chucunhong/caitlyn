#!/usr/bin/env bash
# CAITLYN v2 on both backbones, then the LLM baselines; sequential (the plan allows 3 concurrent calls).
# Usage: bash scripts/run_all_v2.sh <python>
PY="$1"
cd "$(dirname "$0")/.."
mkdir -p results/_logs
export PYTHONWARNINGS=ignore
P=results/_logs/v2_progress.log
for MODEL in default deepseek-v4.1-flash; do
  for DS in german credit_scoring adult; do
    echo "[$(date '+%F %T')] start caitlyn $MODEL $DS" >> $P
    if [ "$MODEL" = default ]; then ARGS=""; else ARGS="--model $MODEL"; fi
    "$PY" -u scripts/run_caitlyn.py --dataset "$DS" $ARGS > "results/_logs/v2_${MODEL}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] done  caitlyn $MODEL $DS (exit $?)" >> $P
  done
done
for M in structsynth cllm; do
  for DS in german credit_scoring adult; do
    echo "[$(date '+%F %T')] start $M $DS" >> $P
    "$PY" -u scripts/run_baseline.py --dataset "$DS" --method "$M" > "results/_logs/bl_${M}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] done  $M $DS (exit $?)" >> $P
  done
done
echo "[$(date '+%F %T')] ALL DONE" >> $P
