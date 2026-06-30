# SLM–RAG Distraction Audit

Code, data, and results for a full-factorial audit of Retrieval-Augmented
Generation (RAG) on six small language models (360M–3.8B parameters), measuring
when retrieved context helps versus *distracts* small models.

Six SLMs × three QA benchmarks (NQ-open, TriviaQA, HotpotQA-distractor) ×
three chunk sizes (128/256/512) × two retrievers (BM25, dense) + closed-book
baselines = **126 main configurations**, plus a **top-k ablation** (k ∈ {1,3,5}),
each over 300 questions (~54,000 inference calls total). We introduce the
**Distraction Ratio (DR)** and use answer **recall@k** to separate retrieval
failure from context-utilisation failure.

## Repository layout

| Path | Contents |
|------|----------|
| `config.py` | Models, datasets, paths, device, token loading |
| `data_prep.py` | Dataset download + per-question corpus construction |
| `chunker.py` | Token-level fixed-size chunking (tiktoken `cl100k_base`) |
| `retriever.py` | BM25 (sparse) and dense (`all-MiniLM-L6-v2`) retrievers |
| `model_runner.py` | SLM loading + generation (Qwen3 thinking suppression) |
| `metrics.py` | Exact Match, token-F1, Distraction Ratio |
| `experiment.py` | Main 126-configuration grid driver |
| `k_ablation_local.py` | Local top-k ablation runner (resume/checkpoint) |
| `compile_k_ablation.py` | Consolidates ablation runs into one table |
| `retrieval_quality.py` | recall@k computation (no LLM, CPU-only) |
| `significance.py` | Wilson 95% CI significance tests |
| `visualize.py`, `visualize_k_ablation.py` | Figures and heatmaps |
| `*.ipynb` | Colab notebooks (gated-model and Phi-3.5 runs) |
| `data/` | Per-question evaluation corpora (JSON) |
| `results/` | Output CSVs and figures |

## Key result files (`results/`)

- `results.csv` — EM, F1, DR for all 126 main configurations
- `k_ablation.csv`, `k_ablation_summary.csv` — top-k ablation
- `retrieval_quality.csv`, `retrieval_vs_em.csv` — recall@k and the utilisation gap
- `significance_ci.csv`, `significance_rag_vs_baseline.csv` — Wilson CI tests

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your HuggingFace token for gated models
```

## Reproduce

```bash
python experiment.py            # main 126-config grid
python k_ablation_local.py      # top-k ablation
python retrieval_quality.py     # recall@k
python significance.py          # Wilson CI tests
python visualize.py             # figures
python visualize_k_ablation.py
```

Apple Silicon (MPS) and single-GPU (Colab T4) are both supported; dense
retrieval runs on CPU. Gated models (Gemma-2, Llama-3.2) require an `HF_TOKEN`.
