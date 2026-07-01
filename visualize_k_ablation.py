"""
Visualize the k ablation: EM vs. top_k, one panel per dataset, one line per model.

Fixed setting: chunk_size=256, retriever=faiss, k ∈ {1,3,5}.
The no_rag baseline (top_k=0) is drawn as a dashed horizontal reference per model.

Usage:
    python visualize_k_ablation.py            # reads results/k_ablation.csv

Outputs:
    results/k_ablation_lineplot.png           # 3-panel EM vs k
    results/k_ablation_summary.csv            # best-k per model×dataset table
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).parent
CSV     = PROJECT / "results" / "k_ablation.csv"
OUT_PNG = PROJECT / "results" / "k_ablation_lineplot.png"
OUT_CSV = PROJECT / "results" / "k_ablation_summary.csv"

MODEL_ORDER = ["SmolLM2-360M", "Qwen3-0.6B", "Gemma-2-2B",
               "Qwen3-1.7B", "Llama-3.2-1B", "Phi-3.5-mini"]
DATASET_ORDER = ["nq", "triviaqa", "hotpotqa"]
DATASET_TITLE = {"nq": "NQ-open", "triviaqa": "TriviaQA", "hotpotqa": "HotpotQA"}
K_VALUES = [1, 3, 5]

COLORS = {
    "SmolLM2-360M": "#888888",
    "Qwen3-0.6B":   "#1f77b4",
    "Gemma-2-2B":   "#2ca02c",
    "Qwen3-1.7B":   "#9467bd",
    "Llama-3.2-1B": "#ff7f0e",
    "Phi-3.5-mini": "#d62728",
}


def main() -> None:
    df = pd.read_csv(CSV)
    rag = df[df["top_k"] > 0].copy()
    baselines = (df[df["top_k"] == 0]
                 .set_index(["dataset", "model"])["em"].to_dict())

    fig, axes = plt.subplots(1, len(DATASET_ORDER),
                             figsize=(16, 5), sharey=False)

    for ax, ds in zip(axes, DATASET_ORDER):
        sub = rag[rag["dataset"] == ds]
        for model in MODEL_ORDER:
            mrows = (sub[sub["model"] == model]
                     .sort_values("top_k"))
            if mrows.empty:
                continue
            ks  = mrows["top_k"].tolist()
            ems = mrows["em"].tolist()
            ax.plot(ks, ems, marker="o", linewidth=2,
                    color=COLORS[model], label=model, zorder=3)

            base = baselines.get((ds, model))
            if base is not None:
                ax.axhline(base, color=COLORS[model], linestyle=":",
                           linewidth=1, alpha=0.4, zorder=1)

        ax.set_title(DATASET_TITLE[ds], fontsize=13, fontweight="bold")
        ax.set_xlabel("top_k (retrieved chunks)", fontsize=11)
        ax.set_xticks(K_VALUES)
        ax.grid(True, alpha=0.3, zorder=0)

    axes[0].set_ylabel("Exact Match (EM)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("k Ablation: Exact Match vs. top_k  "
                 "(chunk_size=256, dense/FAISS retrieval; dotted = no-RAG baseline)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[k-viz] Saved figure → {OUT_PNG}")

    rows = []
    for ds in DATASET_ORDER:
        for model in MODEL_ORDER:
            mrows = rag[(rag["dataset"] == ds) & (rag["model"] == model)]
            if mrows.empty:
                continue
            best = mrows.loc[mrows["em"].idxmax()]
            em_by_k = {int(r["top_k"]): r["em"] for _, r in mrows.iterrows()}
            rows.append({
                "dataset": ds,
                "model": model,
                "best_k": int(best["top_k"]),
                "best_em": round(float(best["em"]), 4),
                "em_k1": em_by_k.get(1),
                "em_k3": em_by_k.get(3),
                "em_k5": em_by_k.get(5),
                "baseline": baselines.get((ds, model)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)
    print(f"[k-viz] Saved summary → {OUT_CSV}\n")

    print("Best-k distribution:")
    print(summary["best_k"].value_counts().sort_index().to_string())
    print("\nPer-dataset best-k:")
    for ds in DATASET_ORDER:
        sub = summary[summary["dataset"] == ds]
        counts = sub["best_k"].value_counts().sort_index().to_dict()
        print(f"  {DATASET_TITLE[ds]:<10}: {counts}")


if __name__ == "__main__":
    main()
