"""Central configuration for the Chunk × SLM Distraction Audit."""
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import torch

MODELS = {
    "SmolLM2-360M": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "dtype": torch.float16,
        "thinking": False,
    },
    "Qwen3-0.6B": {
        "model_id": "Qwen/Qwen3-0.6B",
        "dtype": torch.float16,
        "thinking": True,
    },
    "Gemma-2-2B": {
        "model_id": "google/gemma-2-2b-it",
        "dtype": torch.bfloat16,
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
}

DATASETS    = ["nq", "triviaqa", "hotpotqa"]
CHUNK_SIZES = [128, 256, 512]
RETRIEVERS  = ["bm25", "faiss"]
TOP_K       = 3
NUM_SAMPLES = 300

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR   = BASE_DIR / "cache"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
