#!/usr/bin/env bash
# Re-run Stage D (label-preserving novelty) for Adult and Credit after the baseline chain finishes.
PY="$1"
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
P=results/_logs/v2_progress.log
until grep -q "ALL DONE" $P; do sleep 60; done
for MODEL in default deepseek-v4.1-flash; do
  for DS in adult credit_scoring; do
    echo "[$(date '+%F %T')] start rerun $MODEL $DS" >> $P
    if [ "$MODEL" = default ]; then ARGS=""; else ARGS="--model $MODEL"; fi
    "$PY" -u scripts/run_caitlyn.py --dataset "$DS" $ARGS > "results/_logs/v2r_${MODEL}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] done  rerun $MODEL $DS (exit $?)" >> $P
  done
done
echo "[$(date '+%F %T')] RERUN DONE" >> $P
