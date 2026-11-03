#!/usr/bin/env bash
# CPU baselines (no LLM calls): bash scripts/run_gan_baselines.sh <python> <method>
PY="$1"; M="$2"
cd "$(dirname "$0")/.."
mkdir -p results/_logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for DS in german credit_scoring adult; do
  echo "[$(date '+%F %T')] start $M $DS" >> results/_logs/baselines_progress.log
  "$PY" scripts/run_baseline.py --dataset "$DS" --method "$M" > "results/_logs/bl_${M}_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] done  $M $DS (exit $?)" >> results/_logs/baselines_progress.log
done
