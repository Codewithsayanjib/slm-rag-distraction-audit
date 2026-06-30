#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot launcher for the Chunk × SLM Distraction Audit
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -- 1. Virtual environment --------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[setup] Creating virtual environment …"
    python3 -m venv .venv
fi
source .venv/bin/activate

# -- 2. Dependencies ---------------------------------------------------------
echo "[setup] Installing dependencies …"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# -- 3. HuggingFace token (required for Gemma-2) ----------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "WARNING: HF_TOKEN is not set."
    echo "  Gemma-2-2B requires a HuggingFace token with accepted terms."
    echo "  Export it:  export HF_TOKEN=hf_..."
    echo "  The experiment will attempt to continue; Gemma-2-2B will fail"
    echo "  if the token is missing."
    echo ""
fi

# -- 4. Data preparation -----------------------------------------------------
echo "[step 1/3] Preparing dataset …"
python data_prep.py

# -- 5. Experiment -----------------------------------------------------------
echo "[step 2/3] Running experiment …"
python experiment.py

# -- 6. Visualisation --------------------------------------------------------
echo "[step 3/3] Generating heatmaps …"
python visualize.py

echo ""
echo "Done. Results in: $SCRIPT_DIR/results/"
