#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "[setup] Creating virtual environment …"
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "[setup] Installing dependencies …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "WARNING: HF_TOKEN is not set."
    echo "  Gemma-2-2B requires a HuggingFace token with accepted terms."
    echo "  Export it:  export HF_TOKEN=hf_..."
    echo "  The experiment will attempt to continue; Gemma-2-2B will fail"
    echo "  if the token is missing."
    echo ""
fi

echo "[step 1/3] Preparing dataset …"
python data_prep.py

echo "[step 2/3] Running experiment …"
python experiment.py

echo "[step 3/3] Generating heatmaps …"
python visualize.py

echo ""
echo "Done. Results in: $SCRIPT_DIR/results/"
