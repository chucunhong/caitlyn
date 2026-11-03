#!/usr/bin/env bash
# LLM baselines on the main backbone, sequential (the plan allows 3 concurrent calls in total).
PY="$1"
cd "$(dirname "$0")/.."
mkdir -p results/_logs
export PYTHONWARNINGS=ignore
for M in structsynth cllm; do
  for DS in german credit_scoring adult; do
    echo "[$(date '+%F %T')] start $M $DS" >> results/_logs/baselines_progress.log
    "$PY" -u scripts/run_baseline.py --dataset "$DS" --method "$M" > "results/_logs/bl_${M}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] done  $M $DS (exit $?)" >> results/_logs/baselines_progress.log
  done
done
