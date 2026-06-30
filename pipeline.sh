#!/usr/bin/env bash
# Full pipeline with wall-clock timestamps logged to pipeline.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LOG="$SCRIPT_DIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date "+%Y-%m-%d %H:%M:%S"; }
section() { echo ""; echo "══════════════════════════════════════════════════════"; echo "$(ts)  $1"; echo "══════════════════════════════════════════════════════"; }

section "PIPELINE START"

# ── venv ─────────────────────────────────────────────────────────────────────
section "PHASE 0 — venv + dependencies"
T0=$SECONDS
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "$(ts)  deps done  ($(( SECONDS - T0 ))s)"

# ── data prep ────────────────────────────────────────────────────────────────
section "PHASE 1 — data preparation (NQ-open + Wikipedia)"
T1=$SECONDS
python data_prep.py
echo "$(ts)  data_prep done  ($(( SECONDS - T1 ))s)"

# ── experiment ───────────────────────────────────────────────────────────────
section "PHASE 2 — experiment  (4 models × 4 chunk sizes × 2 retrievers)"
T2=$SECONDS
python experiment.py
echo "$(ts)  experiment done  ($(( SECONDS - T2 ))s)"

# ── visualise ────────────────────────────────────────────────────────────────
section "PHASE 3 — visualisation"
T3=$SECONDS
python visualize.py
echo "$(ts)  visualize done  ($(( SECONDS - T3 ))s)"

section "PIPELINE COMPLETE  (total $(( SECONDS - T0 ))s)"
echo "Results → $SCRIPT_DIR/results/"
