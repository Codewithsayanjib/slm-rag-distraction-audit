#!/usr/bin/env bash

set -u

DATASET="${1:-hotpotqa}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/logs/run_${DATASET}.log"
VENV="$DIR/.venv/bin/activate"

cd "$DIR"
source "$VENV"
mkdir -p logs

MAX_CRASHES=10
MAX_TOTAL_RUNS=300
crashes=0
runs=0

echo "[watchdog] Starting subprocess-per-config mode for dataset=$DATASET" | tee -a "$LOG"

while [ $runs -lt $MAX_TOTAL_RUNS ]; do
    runs=$((runs + 1))
    echo "" | tee -a "$LOG"
    echo "[watchdog] === Run $runs (consecutive crashes: $crashes) ===" | tee -a "$LOG"

    python experiment.py --dataset "$DATASET" --single-config >> "$LOG" 2>&1
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
        echo "[watchdog] Grid complete for $DATASET." | tee -a "$LOG"
        exit 0
    elif [ $EXIT -eq 99 ]; then
        echo "[watchdog] Config completed (exit 99). Restarting for next …" | tee -a "$LOG"
        crashes=0
        sleep 2
    else
        crashes=$((crashes + 1))
        echo "[watchdog] CRASH (exit $EXIT). Consecutive crashes: $crashes / $MAX_CRASHES" | tee -a "$LOG"
        if [ $crashes -ge $MAX_CRASHES ]; then
            echo "[watchdog] Too many consecutive crashes. Giving up." | tee -a "$LOG"
            exit 1
        fi
        sleep 10
    fi
done

echo "[watchdog] Hit MAX_TOTAL_RUNS=$MAX_TOTAL_RUNS. Check $LOG." | tee -a "$LOG"
exit 2
