"""Generate heatmaps and summary tables from results.csv.

With multi-dataset support: one set of heatmaps per dataset, plus a
cross-dataset EM comparison table at the end.

Usage:
    python visualize.py                        # reads results/results.csv
    python visualize.py results/results.csv    # explicit path
    python visualize.py results/results.csv nq # single dataset only
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHUNK_SIZES, DATASETS, MODELS, RESULTS_DIR

# Preferred display order — config.py may have some models commented out (e.g. Phi-3.5-mini
# runs on Colab), so we build the order from config first, then append any extra models
# found in the CSV that aren't in config.
_CONFIG_ORDER = list(MODELS.keys())
MODEL_ORDER = _CONFIG_ORDER  # updated after CSV is loaded (see main block)


# ── core plotting ─────────────────────────────────────────────────────────────

def _matrix(df: pd.DataFrame, model_order: list[str],
            chunk_sizes: list[int], metric: str) -> np.ndarray:
    mat = np.full((len(model_order), len(chunk_sizes)), np.nan)
    for i, m in enumerate(model_order):
        for j, cs in enumerate(chunk_sizes):
            sel = df[(df["model"] == m) & (df["chunk_size"] == cs)][metric]
            if not sel.empty:
                mat[i, j] = sel.values[0]
    return mat


def _heatmap(ax: plt.Axes, mat: np.ndarray, row_labels: list[str],
             col_labels: list[str], title: str,
             vmin: float, vmax: float, cmap: str) -> None:
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Chunk size (tokens)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if (v < vmin + 0.25 * (vmax - vmin) or
                                    v > vmin + 0.75 * (vmax - vmin)) else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ── figure factories ──────────────────────────────────────────────────────────

def plot_metric_grid(df: pd.DataFrame, retriever: str, metric: str,
                     dataset_name: str, out_dir: Path) -> None:
    sub    = df[df["retriever"] == retriever]
    mat    = _matrix(sub, MODEL_ORDER, CHUNK_SIZES, metric)
    labels = {"em": "Exact Match (EM)", "f1": "Token F1",
              "distraction_ratio": "Distraction Ratio"}
    cmaps  = {"em": "YlGn", "f1": "YlGn", "distraction_ratio": "YlOrRd"}
    vmin   = 0.0
    vmax   = 1.0 if metric in ("em", "distraction_ratio") else float(np.nanmax(mat) or 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heatmap(ax, mat, MODEL_ORDER, [str(c) for c in CHUNK_SIZES],
             f"[{dataset_name.upper()}] {labels[metric]} — {retriever.upper()} retrieval",
             vmin, vmax, cmaps[metric])
    plt.tight_layout()
    p = out_dir / f"{dataset_name}_heatmap_{retriever}_{metric}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


def plot_rag_gain(df: pd.DataFrame, retriever: str,
                  dataset_name: str, out_dir: Path) -> None:
    """Signed EM gain vs. no-RAG baseline (blue=helped, red=hurt)."""
    baselines = (df[df["retriever"] == "no_rag"]
                 .set_index("model")["em"].to_dict())
    sub  = df[df["retriever"] == retriever].copy()
    sub["rag_gain"] = sub.apply(
        lambda r: float(r["em"]) - baselines.get(r["model"], 0.0), axis=1)

    mat  = _matrix(sub, MODEL_ORDER, CHUNK_SIZES, "rag_gain")
    vmax = max(abs(float(np.nanmax(mat))), abs(float(np.nanmin(mat))), 0.01)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _heatmap(ax, mat, MODEL_ORDER, [str(c) for c in CHUNK_SIZES],
             f"[{dataset_name.upper()}] RAG EM gain vs. no-RAG — {retriever.upper()} (↑ blue, ↓ red)",
             -vmax, vmax, "RdBu")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.texts[i * len(CHUNK_SIZES) + j].set_text(f"{v:+.3f}")
    plt.tight_layout()
    p = out_dir / f"{dataset_name}_heatmap_{retriever}_rag_gain.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── cross-dataset comparison plot ─────────────────────────────────────────────

def plot_cross_dataset_em(df: pd.DataFrame, out_dir: Path) -> None:
    """3-panel figure: EM by chunk size for each dataset (best retriever per cell)."""
    available = [d for d in DATASETS if d in df["dataset"].unique()]
    if len(available) < 2:
        return   # nothing to compare yet

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5),
                             sharey=True)
    if len(available) == 1:
        axes = [axes]

    rag_df = df[~df["retriever"].isin(["no_rag"])]

    for ax, ds_name in zip(axes, available):
        sub = rag_df[rag_df["dataset"] == ds_name]
        mat = np.full((len(MODEL_ORDER), len(CHUNK_SIZES)), np.nan)
        for i, m in enumerate(MODEL_ORDER):
            for j, cs in enumerate(CHUNK_SIZES):
                vals = sub[(sub["model"] == m) & (sub["chunk_size"] == cs)]["em"]
                if not vals.empty:
                    mat[i, j] = float(vals.max())

        _heatmap(ax, mat, MODEL_ORDER, [str(c) for c in CHUNK_SIZES],
                 f"{ds_name.upper()} — Best EM", 0.0, 1.0, "YlGn")

    fig.suptitle("Cross-dataset EM comparison (best retriever per cell)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = out_dir / "cross_dataset_em.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ── terminal tables ───────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame, dataset_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"[{dataset_name.upper()}] EM by model × chunk size  (best of BM25/FAISS)")
    print("="*70)

    baselines = df[df["retriever"] == "no_rag"].set_index("model")["em"].to_dict()
    rag_df    = df[df["retriever"] != "no_rag"]

    header = f"{'Model':<20}" + "".join(f"  cs={c:<6}" for c in CHUNK_SIZES) + "  base"
    print(header)
    print("-" * len(header))
    for m in MODEL_ORDER:
        row_str = f"{m:<20}"
        for cs in CHUNK_SIZES:
            vals = rag_df[(rag_df["model"] == m) & (rag_df["chunk_size"] == cs)]["em"]
            best = float(vals.max()) if not vals.empty else float("nan")
            row_str += f"  {best:.3f}  "
        row_str += f"  {baselines.get(m, float('nan')):.3f}"
        print(row_str)

    print()
    print("Distraction Ratio (0=no distraction, 1=fully distracted):")
    print("-" * len(header))
    for m in MODEL_ORDER:
        row_str = f"{m:<20}"
        for cs in CHUNK_SIZES:
            vals = rag_df[(rag_df["model"] == m) & (rag_df["chunk_size"] == cs)]["distraction_ratio"]
            worst = float(vals.max()) if not vals.empty else float("nan")
            row_str += f"  {worst:.3f}  "
        print(row_str)
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main(results_path: Path | None = None,
         only_dataset: str | None = None) -> None:
    if results_path is None:
        results_path = RESULTS_DIR / "results.csv"
    if not results_path.exists():
        results_path = RESULTS_DIR / "results_partial.csv"

    print(f"[visualize] Reading {results_path}")
    df = pd.read_csv(results_path)
    df["chunk_size"] = pd.to_numeric(df["chunk_size"], errors="coerce")

    # Back-compat: if no dataset column, treat everything as nq
    if "dataset" not in df.columns:
        df["dataset"] = "nq"

    # Rebuild MODEL_ORDER to include any models in the CSV not in config (e.g. Phi-3.5-mini
    # which is commented out of config.py but runs on Colab and gets merged in later).
    global MODEL_ORDER
    csv_models = df["model"].unique().tolist()
    extra = [m for m in csv_models if m not in _CONFIG_ORDER]
    MODEL_ORDER = _CONFIG_ORDER + extra

    out = results_path.parent
    available = df["dataset"].unique().tolist()
    targets   = [only_dataset] if only_dataset else available

    print(f"[visualize] Datasets present: {available}")
    print("[visualize] Generating heatmaps …")

    for ds_name in targets:
        ds_df = df[df["dataset"] == ds_name]
        if ds_df.empty:
            print(f"  [skip] no data for {ds_name}")
            continue

        for retriever in ["bm25", "faiss"]:
            for metric in ["em", "f1", "distraction_ratio"]:
                plot_metric_grid(ds_df, retriever, metric, ds_name, out)
            plot_rag_gain(ds_df, retriever, ds_name, out)

        print_summary_table(ds_df, ds_name)

    # Cross-dataset comparison (only if multiple datasets present)
    plot_cross_dataset_em(df, out)

    print(f"[visualize] All figures written to {out}/")


if __name__ == "__main__":
    path    = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ds_only = sys.argv[2] if len(sys.argv) > 2 else None
    main(path, ds_only)
