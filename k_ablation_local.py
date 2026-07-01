"""
k_ablation_local.py — runs the 8 missing k-ablation configs locally.

Targets (chunk_size=256, retriever=faiss, all fixed):
  Qwen3-0.6B  × {nq, triviaqa, hotpotqa} × k = {1, 5}   → 6 configs
  Qwen3-1.7B  × hotpotqa                 × k = {1, 5}   → 2 configs

Uses the same pipeline as experiment.py (model_runner.py handles
enable_thinking=False properly for both Qwen3 models — no Colab hacks needed).

Usage:
    python k_ablation_local.py           # full run  (300 samples each)
    python k_ablation_local.py --smoke   # 3 samples per config — verify only
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from chunker import chunk_text
from config import DATA_DIR, DEVICE, RESULTS_DIR
from data_prep import prepare_dataset
from metrics import distraction_ratio, exact_match, token_f1
from model_runner import generate_answer, load_model
from retriever import build_retriever


TARGETS = [
    ("Qwen3-0.6B",  "nq"),
    ("Qwen3-0.6B",  "triviaqa"),
    ("Qwen3-0.6B",  "hotpotqa"),
    ("Qwen3-1.7B",  "hotpotqa"),
]
TOP_K_VALUES   = [1, 5]
CHUNK_SIZE     = 256
RETRIEVER_TYPE = "faiss"
NUM_SAMPLES    = 300

BASELINE_EM = {
    ("Qwen3-0.6B", "nq"):       0.0267,
    ("Qwen3-0.6B", "triviaqa"): 0.0833,
    ("Qwen3-0.6B", "hotpotqa"): 0.0533,
    ("Qwen3-1.7B", "hotpotqa"): 0.0433,
}

OUT_CSV = RESULTS_DIR / "k_ablation_local.csv"


def _flush() -> None:
    gc.collect()
    if DEVICE == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_done() -> set[tuple]:
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        return {(r["model"], r["dataset"], int(r["top_k"]))
                for _, r in df.iterrows()}
    return set()


def _append_row(row: dict) -> None:
    df_new = pd.DataFrame([row])
    if OUT_CSV.exists():
        df_new.to_csv(OUT_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(OUT_CSV, index=False)



def run_config(model_name: str, dataset_name: str,
               samples: list[dict], top_k: int,
               retriever, smoke: bool) -> dict:

    base_em = BASELINE_EM[(model_name, dataset_name)]
    em_list, f1_list = [], []

    desc = f"  [faiss k={top_k}] {model_name} {dataset_name}"
    for idx, s in enumerate(tqdm(samples, desc=desc, leave=False)):
        try:
            context = "\n\n".join(retriever.retrieve(s["question"], top_k=top_k))
            pred    = generate_answer(model_name, s["question"], context=context)
        except Exception as exc:
            print(f"\n  [WARN] sample {s['id']} failed: {exc}")
            pred = ""

        em  = exact_match(pred, s["answers"])
        f1  = token_f1(pred, s["answers"])
        em_list.append(em)
        f1_list.append(f1)

        if smoke:
            print(f"    Q: {s['question'][:70]}")
            print(f"    A (gold): {s['answers'][:3]}  |  pred: '{pred}'  EM={em}")

        if (idx + 1) % 10 == 0:
            _flush()

    avg_em = sum(em_list) / len(em_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    return {
        "dataset":           dataset_name,
        "model":             model_name,
        "chunk_size":        CHUNK_SIZE,
        "retriever":         RETRIEVER_TYPE,
        "top_k":             top_k,
        "em":                round(avg_em, 4),
        "f1":                round(avg_f1, 4),
        "distraction_ratio": round(distraction_ratio(base_em, avg_em), 4),
        "n_samples":         len(samples),
    }



def main(smoke: bool = False) -> None:
    n = 3 if smoke else NUM_SAMPLES
    mode = f"SMOKE ({n} samples)" if smoke else f"FULL ({n} samples)"
    print(f"\n{'═'*60}")
    print(f"k_ablation_local.py  —  {mode}")
    print(f"{'═'*60}")

    RESULTS_DIR.mkdir(exist_ok=True)
    done = _load_done()

    for model_name, dataset_name in TARGETS:
        print(f"\n── {model_name}  ×  {dataset_name} ──")

        samples = prepare_dataset(dataset_name)[:n]

        all_chunks: list[str] = []
        for s in samples:
            all_chunks.extend(chunk_text(s["document"], CHUNK_SIZE))
        print(f"  [index] faiss | cs={CHUNK_SIZE} | {len(all_chunks)} chunks")
        retriever = build_retriever(all_chunks, RETRIEVER_TYPE)

        load_model(model_name)

        for top_k in TOP_K_VALUES:
            key = (model_name, dataset_name, top_k)
            if key in done:
                print(f"  [skip] {model_name} {dataset_name} k={top_k} already done")
                continue

            row = run_config(model_name, dataset_name, samples,
                             top_k, retriever, smoke)
            print(f"  k={top_k}  EM={row['em']:.4f}  F1={row['f1']:.4f}  "
                  f"DR={row['distraction_ratio']:.4f}")

            if not smoke:
                _append_row(row)
                done.add(key)

        del retriever, all_chunks
        _flush()

    if smoke:
        print(f"\n{'═'*60}")
        print("SMOKE TEST PASSED — pipeline works. Results NOT saved.")
        print("Run without --smoke for the full experiment.")
        print(f"{'═'*60}")
    else:
        df = pd.read_csv(OUT_CSV)
        print(f"\n[done] {len(df)} rows saved → {OUT_CSV}")
        print("Next: run compile_k_ablation.py to merge into k_ablation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run 3 samples per config to verify pipeline only")
    args = parser.parse_args()
    main(smoke=args.smoke)
