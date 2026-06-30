"""
Statistical significance for EM scores using Wilson binomial confidence intervals.
No new model runs needed — works entirely from results.csv.

Usage:
    python significance.py                     # reads results/results.csv
    python significance.py results/results.csv

Outputs:
  1. Per-config CIs printed to terminal
  2. results/significance_ci.csv  — full table with lower/upper bounds
  3. results/significance_rag_vs_baseline.csv — pairwise RAG vs no-RAG tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── McNemar-style approximation via CIs ──────────────────────────────────────

def cis_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return lo1 <= hi2 and lo2 <= hi1


def sig_label(p_lo: float, p_hi: float,
              q_lo: float, q_hi: float) -> str:
    """Return significance label based on CI non-overlap."""
    if cis_overlap(p_lo, p_hi, q_lo, q_hi):
        return "ns"       # not significant
    if p_lo > q_hi:
        return "↑ sig"    # first group significantly higher
    return "↓ sig"        # first group significantly lower


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_path: Path | None = None) -> None:
    if results_path is None:
        results_path = Path(__file__).parent / "results" / "results.csv"

    df = pd.read_csv(results_path)
    df["chunk_size"] = df["chunk_size"].astype(str)
    n_col = "n_samples" if "n_samples" in df.columns else None

    # ── Step 1: Compute CI for every row ─────────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        p = float(row["em"])
        n = int(row[n_col]) if n_col else 300
        lo, hi = wilson_ci(p, n)
        rows.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "chunk_size": row["chunk_size"],
            "retriever": row["retriever"],
            "em": p,
            "n": n,
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_width": round(hi - lo, 4),
        })

    ci_df = pd.DataFrame(rows)
    out_dir = results_path.parent
    ci_df.to_csv(out_dir / "significance_ci.csv", index=False)
    print(f"[significance] Saved per-config CIs → {out_dir}/significance_ci.csv")

    # ── Step 2: RAG vs no-RAG pairwise significance ───────────────────────────
    baseline_rows = ci_df[ci_df["retriever"] == "no_rag"].set_index(
        ["dataset", "model"]
    )[["em", "ci_lo", "ci_hi"]]

    rag_rows = ci_df[ci_df["retriever"] != "no_rag"].copy()

    pairs = []
    for _, r in rag_rows.iterrows():
        key = (r["dataset"], r["model"])
        if key not in baseline_rows.index:
            continue
        base = baseline_rows.loc[key]
        sig = sig_label(r["ci_lo"], r["ci_hi"],
                        float(base["ci_lo"]), float(base["ci_hi"]))
        delta = round(r["em"] - float(base["em"]), 4)
        pairs.append({
            "dataset": r["dataset"],
            "model": r["model"],
            "retriever": r["retriever"],
            "chunk_size": r["chunk_size"],
            "baseline_em": round(float(base["em"]), 4),
            "rag_em": round(r["em"], 4),
            "delta_em": delta,
            "baseline_ci": f"[{base['ci_lo']:.3f}, {base['ci_hi']:.3f}]",
            "rag_ci": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
            "significance": sig,
        })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(out_dir / "significance_rag_vs_baseline.csv", index=False)
    print(f"[significance] Saved RAG-vs-baseline tests → {out_dir}/significance_rag_vs_baseline.csv\n")

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    total = len(pairs_df)
    sig_up = (pairs_df["significance"] == "↑ sig").sum()
    sig_dn = (pairs_df["significance"] == "↓ sig").sum()
    ns     = (pairs_df["significance"] == "ns").sum()

    print(f"{'='*65}")
    print(f"SIGNIFICANCE SUMMARY  (95% Wilson CI, n=300 per config)")
    print(f"{'='*65}")
    print(f"  Total RAG configs tested : {total}")
    print(f"  RAG significantly BETTER : {sig_up}  ({100*sig_up/total:.1f}%)")
    print(f"  RAG significantly WORSE  : {sig_dn}  ({100*sig_dn/total:.1f}%)")
    print(f"  Not significant          : {ns}  ({100*ns/total:.1f}%)")
    print()

    # Per-dataset breakdown
    for ds in sorted(pairs_df["dataset"].unique()):
        sub = pairs_df[pairs_df["dataset"] == ds]
        up = (sub["significance"] == "↑ sig").sum()
        dn = (sub["significance"] == "↓ sig").sum()
        n  = len(sub)
        print(f"  [{ds.upper():<10}]  better={up}/{n}  worse={dn}/{n}  ns={(n-up-dn)}/{n}")

    print()
    print(f"{'='*65}")
    print("NOTABLE SIGNIFICANT DIFFERENCES (|delta| > 0.05, n≥300)")
    print(f"{'='*65}")
    notable = pairs_df[
        (pairs_df["significance"] != "ns") &
        (pairs_df["delta_em"].abs() > 0.05)
    ].sort_values("delta_em", ascending=False)
    if notable.empty:
        print("  None above |delta|=0.05 threshold.")
    else:
        print(notable[["dataset","model","retriever","chunk_size",
                        "baseline_em","rag_em","delta_em","significance"]]
              .to_string(index=False))

    print()
    print(f"{'='*65}")
    print("SIGNIFICANT DEGRADATION (RAG significantly WORSE than baseline)")
    print(f"{'='*65}")
    worse = pairs_df[pairs_df["significance"] == "↓ sig"].sort_values("delta_em")
    if worse.empty:
        print("  None.")
    else:
        print(worse[["dataset","model","retriever","chunk_size",
                      "baseline_em","rag_em","delta_em","significance"]]
              .to_string(index=False))


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(path)
