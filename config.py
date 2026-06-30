"""Central configuration for the Chunk × SLM Distraction Audit."""
import os
from pathlib import Path

# Prevent HuggingFace tokenizer workers from forking — they corrupt the MPS
# context on macOS when sentence-transformers is also in the process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load .env if present (keeps the token out of env but available to all modules)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import torch

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = {
    "SmolLM2-360M": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "dtype": torch.float16,
        "thinking": False,
    },
    "Qwen3-0.6B": {
        "model_id": "Qwen/Qwen3-0.6B",
        "dtype": torch.float16,
        "thinking": True,   # supports /nothink
    },
    "Gemma-2-2B": {
        "model_id": "google/gemma-2-2b-it",
        "dtype": torch.bfloat16,  # Gemma-2 was trained in bfloat16
        "thinking": False,
    },
    "Qwen3-1.7B": {
        "model_id": "Qwen/Qwen3-1.7B",
        "dtype": torch.float16,
        "thinking": True,
    },
    "Llama-3.2-1B": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "dtype": torch.bfloat16,
        "thinking": False,
    },
    # "Phi-3.5-mini": {                      # runs on Colab — not locally
    #     "model_id": "microsoft/Phi-3.5-mini-instruct",
    #     "dtype": torch.bfloat16,
    #     "thinking": False,
    # },
}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
DATASETS    = ["nq", "triviaqa", "hotpotqa"]   # outer loop
CHUNK_SIZES = [128, 256, 512]                   # tokens (tiktoken cl100k_base); 1024 removed — context overflow artifact
RETRIEVERS  = ["bm25", "faiss"]
TOP_K       = 3                                 # chunks returned per query
NUM_SAMPLES = 300

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR   = BASE_DIR / "cache"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# HuggingFace token (needed for Gemma-2)
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
