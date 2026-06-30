"""
Compile clean k_ablation.csv from four sources:

  Source A: ~/Downloads/k_ablation_partial (1).csv
            → k=1,5 for NQ + TriviaQA (5 valid models, Qwen3-0.6B excluded)

  Source B: ~/Downloads/k_ablation_remaining.csv
            → k=1,5 for HotpotQA (SmolLM2, Gemma, Llama, Phi only)

  Source C: results/results.csv (cs=256, faiss)
            → k=3 for all 6 models × 3 datasets (reliable local run)
            → no_rag baseline for all 6 models × 3 datasets

  Source D: results/k_ablation_local.csv
            → k=1,5 for Qwen3-0.6B × all 3 datasets (local fix run)
            → k=1,5 for Qwen3-1.7B × HotpotQA (local fix run)

Output: results/k_ablation.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
PROJECT   = Path(__file__).parent
MAIN_CSV  = PROJECT / "results" / "results.csv"
OUT_CSV   = PROJECT / "results" / "k_ablation.csv"

# ── Load sources ──────────────────────────────────────────────────────────────

partial   = pd.read_csv(DOWNLOADS / "k_ablation_partial (1).csv")
remaining = pd.read_csv(DOWNLOADS / "k_ablation_remaining.csv")
main      = pd.read_csv(MAIN_CSV)
local     = pd.read_csv(PROJECT / "results" / "k_ablation_local.csv")

# ── Source A: k=1,5 from NQ + TriviaQA (Colab partial run) ──────────────────
# Keep: k ∈ {1,5}, not no_rag, not Qwen3-0.6B (broken zeros on Colab)
src_a = partial[
    partial["top_k"].isin([1, 5]) &
    (partial["retriever"] != "no_rag") &
    (partial["model"] != "Qwen3-0.6B")
].copy()
print(f"Source A (partial k=1,5 NQ+TriviaQA): {len(src_a)} rows")

# ── Source B: k=1,5 from HotpotQA (Colab remaining run) ─────────────────────
# Keep: valid models only (exclude both Qwen3 — Qwen3-0.6B broken, Qwen3-1.7B zeros)
src_b = remaining[
    remaining["top_k"].isin([1, 5]) &
    (remaining["retriever"] != "no_rag") &
    ~remaining["model"].str.startswith("Qwen3")
].copy()
print(f"Source B (remaining k=1,5 HotpotQA):  {len(src_b)} rows")

# ── Source D: k=1,5 for Qwen3 models from local fix run ──────────────────────
src_d = local[local["top_k"].isin([1, 5])].copy()
src_d["chunk_size"] = src_d["chunk_size"].astype(str)
print(f"Source D (local fix Qwen3 k=1,5):     {len(src_d)} rows")

# ── Source C-1: k=3 from results.csv (cs=256, faiss, all 6 models) ───────────
k3 = main[
    (main["retriever"] == "faiss") &
    (main["chunk_size"].astype(str) == "256")
].copy()
k3["top_k"] = 3
# Ensure column order matches k-ablation files
k3 = k3[["dataset","model","chunk_size","retriever","top_k","em","f1",
          "distraction_ratio","n_samples"]]
print(f"Source C-1 (k=3 from results.csv):     {len(k3)} rows")

# ── Source C-2: no_rag baselines from results.csv ────────────────────────────
no_rag = main[main["retriever"] == "no_rag"].copy()
no_rag["top_k"] = 0
no_rag["chunk_size"] = "none"
no_rag = no_rag[["dataset","model","chunk_size","retriever","top_k","em","f1",
                  "distraction_ratio","n_samples"]]
print(f"Source C-2 (no_rag from results.csv):  {len(no_rag)} rows")

# ── Combine ───────────────────────────────────────────────────────────────────
# Align columns before concat
col_order = ["dataset","model","chunk_size","retriever","top_k","em","f1",
             "distraction_ratio","n_samples"]

for df in [src_a, src_b]:
    df["chunk_size"] = df["chunk_size"].astype(str)

combined = pd.concat(
    [no_rag, src_a[col_order], src_b[col_order], src_d[col_order], k3],
    ignore_index=True
)

# ── Sort ──────────────────────────────────────────────────────────────────────
MODEL_ORDER = ["SmolLM2-360M","Qwen3-0.6B","Gemma-2-2B",
               "Qwen3-1.7B","Llama-3.2-1B","Phi-3.5-mini"]
DATASET_ORDER = ["nq","triviaqa","hotpotqa"]

combined["_ds_rank"]  = combined["dataset"].map({d:i for i,d in enumerate(DATASET_ORDER)})
combined["_mdl_rank"] = combined["model"].map({m:i for i,m in enumerate(MODEL_ORDER)})
combined = (combined
            .sort_values(["_ds_rank","_mdl_rank","top_k"])
            .drop(columns=["_ds_rank","_mdl_rank"])
            .reset_index(drop=True))

combined.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(combined)} rows → {OUT_CSV}")

# ── Coverage report ───────────────────────────────────────────────────────────
print("\n=== COVERAGE (rows per model × dataset × top_k) ===")
rag_only = combined[combined["top_k"] > 0]
pivot = (rag_only
         .groupby(["model","dataset"])["top_k"]
         .apply(lambda s: sorted(s.tolist()))
         .unstack("dataset"))
print(pivot.to_string())

print("\n=== MISSING k=1,5 ENTRIES (Qwen3 known issues) ===")
all_combos = [(m, d, k) for m in MODEL_ORDER for d in DATASET_ORDER for k in [1,5]]
present = set(zip(rag_only["model"], rag_only["dataset"], rag_only["top_k"]))
missing = [(m,d,k) for m,d,k in all_combos if (m,d,k) not in present]
if missing:
    for m,d,k in missing:
        print(f"  MISSING: {m} / {d} / k={k}")
else:
    print("  None")
