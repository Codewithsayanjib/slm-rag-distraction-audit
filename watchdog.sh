#!/usr/bin/env bash
# watchdog.sh — runs experiment.py ONE CONFIG AT A TIME in a fresh subprocess.
#
# Exit codes from experiment.py --single-config:
#   0  → grid complete, stop
#   99 → completed one config, restart for next
#   *  → crash, restart (up to MAX_CRASHES consecutive)
#
# Each config gets a guaranteed fresh MPS state because it runs in its own
# Python process. This eliminates the cumulative MPS memory fragmentation
# that was killing cs=512 on HotpotQA.
#
# Usage: bash watchdog.sh [nq|triviaqa|hotpotqa]

set -u

DATASET="${1:-hotpotqa}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/logs/run_${DATASET}.log"
VENV="$DIR/.venv/bin/activate"

cd "$DIR"
source "$VENV"
mkdir -p logs

MAX_CRASHES=10            # max consecutive non-99 failures before giving up
MAX_TOTAL_RUNS=300         # safety ceiling on total subprocess launches
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
        crashes=0           # reset crash counter on successful completion
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
