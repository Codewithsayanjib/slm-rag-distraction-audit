"""
retrieval_quality.py — measures recall@k (answer-in-context coverage).

Reviewer concern #3: separate RETRIEVAL FAILURE from DISTRACTION.
  - If recall@k is high but EM is low  → the model was distracted (answer was there).
  - If recall@k is low                 → retrieval failed (answer never reached the model).

No LLM inference. CPU-only. Mirrors experiment.py's retrieval exactly:
the chunks of all 300 documents are pooled into one index, then each question
retrieves top-k from that shared pool.

Grid: 3 datasets × 3 chunk_sizes × 2 retrievers = 18 retriever builds.
For each we retrieve top-5 once and derive recall@{1,3,5} from the ranked list.

Usage:
    python retrieval_quality.py            # full run (300 samples each)
    python retrieval_quality.py --smoke    # 5 samples per config — verify only

Outputs:
    results/retrieval_quality.csv          # recall@k per (dataset, cs, retriever)
    results/retrieval_vs_em.csv            # recall@3 vs actual EM@3 — the distraction gap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from chunker import chunk_text
from config import DATASETS, NUM_SAMPLES, RESULTS_DIR
from data_prep import prepare_dataset
from metrics import _normalize
from retriever import build_retriever

CHUNK_SIZES = [128, 256, 512]
RETRIEVERS  = ["bm25", "faiss"]
K_VALUES    = [1, 3, 5]
MAX_K       = max(K_VALUES)

OUT_RECALL  = RESULTS_DIR / "retrieval_quality.csv"
OUT_VS_EM   = RESULTS_DIR / "retrieval_vs_em.csv"

SHORT_ANSWER_CHARS = 3


def _answer_in_text(answers: list[str], text_norm: str) -> bool:
    """True if any gold answer appears as a whitespace-bounded span in text_norm.

    Both sides are normalized with metrics._normalize (lowercase, strip articles
    and punctuation, collapse whitespace) so matching is consistent with EM.
    """
    padded = f" {text_norm} "
    for a in answers:
        a_norm = _normalize(a)
        if a_norm and f" {a_norm} " in padded:
            return True
    return False


def _is_short(answers: list[str]) -> bool:
    return all(len(_normalize(a)) <= SHORT_ANSWER_CHARS for a in answers)


def run_config(dataset_name: str, samples: list[dict], chunk_size: int,
               retriever_type: str) -> dict:
    all_chunks: list[str] = []
    for s in samples:
        all_chunks.extend(chunk_text(s["document"], chunk_size))

    retriever = build_retriever(all_chunks, retriever_type)

    hits = {k: 0 for k in K_VALUES}
    short_count = 0
    desc = f"  {retriever_type} cs={chunk_size}"
    for s in tqdm(samples, desc=desc, leave=False):
        ranked = retriever.retrieve(s["question"], top_k=MAX_K)
        if _is_short(s["answers"]):
            short_count += 1
        for k in K_VALUES:
            ctx_norm = _normalize("\n\n".join(ranked[:k]))
            if _answer_in_text(s["answers"], ctx_norm):
                hits[k] += 1

    n = len(samples)
    return {
        "dataset":      dataset_name,
        "chunk_size":   chunk_size,
        "retriever":    retriever_type,
        "recall_at_1":  round(hits[1] / n, 4),
        "recall_at_3":  round(hits[3] / n, 4),
        "recall_at_5":  round(hits[5] / n, 4),
        "short_answer_frac": round(short_count / n, 4),
        "n_samples":    n,
    }


def main(smoke: bool = False) -> None:
    n = 5 if smoke else NUM_SAMPLES
    mode = f"SMOKE ({n} samples)" if smoke else f"FULL ({n} samples)"
    print(f"\n{'='*60}")
    print(f"retrieval_quality.py — recall@k — {mode}")
    print(f"{'='*60}")

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for ds in DATASETS:
        print(f"\n── {ds.upper()} ──")
        samples = prepare_dataset(ds)[:n]
        for cs in CHUNK_SIZES:
            for ret in RETRIEVERS:
                row = run_config(ds, samples, cs, ret)
                rows.append(row)
                print(f"  cs={cs:4d} {ret:5s}  "
                      f"R@1={row['recall_at_1']:.3f}  "
                      f"R@3={row['recall_at_3']:.3f}  "
                      f"R@5={row['recall_at_5']:.3f}  "
                      f"(short={row['short_answer_frac']:.2f})")

    recall_df = pd.DataFrame(rows)

    if smoke:
        print(f"\n{'='*60}")
        print("SMOKE TEST PASSED — pipeline works. Results NOT saved.")
        print(f"{'='*60}")
        return

    recall_df.to_csv(OUT_RECALL, index=False)
    print(f"\n[recall] Saved → {OUT_RECALL}")

    main_csv = RESULTS_DIR / "results.csv"
    if main_csv.exists():
        em = pd.read_csv(main_csv)
        em = em[em["retriever"].isin(RETRIEVERS)].copy()
        em["chunk_size"] = em["chunk_size"].astype(str)
        recall_df["chunk_size"] = recall_df["chunk_size"].astype(str)

        merged = em.merge(
            recall_df[["dataset", "chunk_size", "retriever", "recall_at_3"]],
            on=["dataset", "chunk_size", "retriever"], how="inner",
        )
        merged["distraction_gap"] = (
            merged["recall_at_3"] - merged["em"]
        ).round(4)
        merged = merged[["dataset", "model", "chunk_size", "retriever",
                         "recall_at_3", "em", "distraction_gap"]]
        merged = merged.sort_values(["dataset", "model", "chunk_size", "retriever"])
        merged.to_csv(OUT_VS_EM, index=False)
        print(f"[gap]    Saved recall@3 vs EM@3 → {OUT_VS_EM}")

        print(f"\n{'='*60}")
        print("RETRIEVAL CEILING vs MODEL EM  (recall@3 averaged over models)")
        print(f"{'='*60}")
        for ds in DATASETS:
            sub = merged[merged["dataset"] == ds]
            r3  = sub["recall_at_3"].mean()
            mean_em = sub["em"].mean()
            print(f"  {ds.upper():<10}  recall@3={r3:.3f}  "
                  f"mean_EM={mean_em:.3f}  gap={r3-mean_em:.3f}")
    else:
        print("[gap]    results.csv not found — skipped recall-vs-EM merge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run 5 samples per config to verify pipeline only")
    args = parser.parse_args()
    main(smoke=args.smoke)
