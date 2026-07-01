"""Main experiment loop — datasets × models × chunk sizes × retrievers × no-RAG baseline.

Grid:
  datasets   : nq, triviaqa, hotpotqa          (outer)
  models     : SmolLM2-360M, Qwen3-0.6B, Gemma-2-2B, Qwen3-1.7B
  chunk sizes: 128, 256, 512, 1024 tokens
  retrievers : bm25, faiss (dense cosine)

Supports resume: completed (dataset, model, chunk_size, retriever) rows are skipped.
Results accumulate in results/results_partial.csv; finalized as results/results.csv.

Usage:
    python experiment.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from chunker import chunk_text
from config import CHUNK_SIZES, DATASETS, DEVICE, MODELS, NUM_SAMPLES, RESULTS_DIR, RETRIEVERS, TOP_K
from data_prep import prepare_dataset
from metrics import distraction_ratio, exact_match, token_f1
from model_runner import generate_answer
from retriever import build_retriever


def _flush_memory() -> None:
    """Aggressively free MPS/CUDA cache and run GC between configs."""
    gc.collect()
    if DEVICE == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()



PARTIAL = RESULTS_DIR / "results_partial.csv"
FINAL   = RESULTS_DIR / "results.csv"



def _migrate_csv(path: Path) -> None:
    """Add 'dataset' column (='nq') to legacy CSVs that predate multi-dataset support."""
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df.insert(0, "dataset", "nq")
        df.to_csv(path, index=False)
        print(f"[migrate] Added 'dataset' column to {path.name}")


def _load_done() -> tuple[set[tuple], list[dict]]:
    """Return (done_keys, existing_rows) merging both results.csv and results_partial.csv.

    Merging ensures we never lose progress when a crash leaves rows split across
    the two files (e.g. previous dataset in results.csv, current dataset in partial).
    """
    frames = []
    for p in (FINAL, PARTIAL):
        if p.exists():
            _migrate_csv(p)
            frames.append(pd.read_csv(p))

    if not frames:
        return set(), []

    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["dataset", "model", "chunk_size", "retriever"]
    )
    keys = {
        (r["dataset"], r["model"], str(r["chunk_size"]), r["retriever"])
        for _, r in df.iterrows()
    }
    return keys, df.to_dict("records")


def _save(results: list[dict]) -> None:
    pd.DataFrame(results).to_csv(PARTIAL, index=False)



def run_baseline(dataset_name: str, model_name: str, samples: list[dict]) -> dict:
    em_list, f1_list = [], []
    for s in tqdm(samples, desc=f"  [no-RAG] {model_name}", leave=False):
        pred = generate_answer(model_name, s["question"], context=None)
        em_list.append(exact_match(pred, s["answers"]))
        f1_list.append(token_f1(pred, s["answers"]))

    return {
        "dataset":           dataset_name,
        "model":             model_name,
        "chunk_size":        "none",
        "retriever":         "no_rag",
        "em":                round(sum(em_list) / len(em_list), 4),
        "f1":                round(sum(f1_list) / len(f1_list), 4),
        "distraction_ratio": 0.0,
        "n_samples":         len(samples),
    }



def run_rag(dataset_name: str, model_name: str, samples: list[dict],
            chunk_size: int, retriever_type: str, baseline_em: float) -> dict:

    _flush_memory()

    all_chunks: list[str] = []
    for s in samples:
        all_chunks.extend(chunk_text(s["document"], chunk_size))

    print(f"  [index] {retriever_type} | cs={chunk_size} | {len(all_chunks)} chunks")
    retriever = build_retriever(all_chunks, retriever_type)

    em_list, f1_list = [], []
    desc = f"  [{retriever_type}] {model_name} cs={chunk_size}"
    for idx, s in enumerate(tqdm(samples, desc=desc, leave=False)):
        try:
            context = "\n\n".join(retriever.retrieve(s["question"], top_k=TOP_K))
            pred    = generate_answer(model_name, s["question"], context=context)
        except Exception as exc:
            print(f"\n  [WARN] inference failed for sample {s['id']}: {exc}", flush=True)
            pred = ""
        em_list.append(exact_match(pred, s["answers"]))
        f1_list.append(token_f1(pred, s["answers"]))
        if (idx + 1) % 10 == 0:
            _flush_memory()

    del retriever, all_chunks
    _flush_memory()

    avg_em = sum(em_list) / len(em_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    return {
        "dataset":           dataset_name,
        "model":             model_name,
        "chunk_size":        chunk_size,
        "retriever":         retriever_type,
        "em":                round(avg_em, 4),
        "f1":                round(avg_f1, 4),
        "distraction_ratio": round(distraction_ratio(baseline_em, avg_em), 4),
        "n_samples":         len(samples),
    }



def run_experiment(only_dataset: str | None = None,
                   single_config: bool = False) -> pd.DataFrame:
    """Run the experiment grid.

    If single_config=True, runs exactly ONE pending config (baseline or RAG)
    and exits with code 99 — so a parent watchdog can spawn each config in
    its own subprocess, guaranteeing fresh MPS state per config. Exit 0
    means everything is already complete.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    done, results = _load_done()

    datasets_to_run = [only_dataset] if only_dataset else DATASETS
    for dataset_name in datasets_to_run:
        print(f"\n{'═'*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'═'*60}")

        samples = prepare_dataset(dataset_name)[:NUM_SAMPLES]

        for model_name in MODELS:
            key_base = (dataset_name, model_name, "none", "no_rag")
            if key_base not in done:
                print(f"\n{'─'*60}")
                print(f"[{dataset_name}] Model: {model_name}")
                row = run_baseline(dataset_name, model_name, samples)
                results.append(row)
                done.add(key_base)
                _save(results)
                _flush_memory()
                print(f"  Baseline  EM={row['em']:.3f}  F1={row['f1']:.3f}")
                if single_config:
                    sys.exit(99)
            else:
                print(f"\n[skip] Baseline already done: [{dataset_name}] {model_name}")

            base_em = next(
                r["em"] for r in results
                if r["dataset"]  == dataset_name
                and r["model"]   == model_name
                and r["retriever"] == "no_rag"
            )

            for cs in CHUNK_SIZES:
                for ret in RETRIEVERS:
                    key = (dataset_name, model_name, str(cs), ret)
                    if key in done:
                        print(f"  [skip] [{dataset_name}] {model_name} cs={cs} {ret}")
                        continue

                    row = run_rag(dataset_name, model_name, samples, cs, ret, base_em)
                    results.append(row)
                    done.add(key)
                    _save(results)
                    _flush_memory()
                    print(f"  [{dataset_name}] cs={cs:4d}  {ret:5s}  "
                          f"EM={row['em']:.3f}  F1={row['f1']:.3f}  "
                          f"DR={row['distraction_ratio']:.3f}")
                    if single_config:
                        sys.exit(99)

    df = pd.DataFrame(results)
    df.to_csv(FINAL, index=False)
    PARTIAL.unlink(missing_ok=True)
    print(f"\n[done] Results saved to {FINAL}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=DATASETS, default=None,
        help="Run only this dataset (default: all in order)",
    )
    parser.add_argument(
        "--single-config", action="store_true",
        help="Run exactly one pending config then exit 99 (for watchdog subprocess mode)",
    )
    args = parser.parse_args()
    run_experiment(only_dataset=args.dataset, single_config=args.single_config)
