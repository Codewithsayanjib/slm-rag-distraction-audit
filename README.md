<!-- ============================ HEADER ============================ -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1F4E79,100:2E75B6&height=190&section=header&text=Found%20but%20Not%20Used&fontSize=42&fontColor=ffffff&fontAlignY=35&desc=A%20Full-Factorial%20Audit%20of%20RAG%20Distraction%20in%20Small%20Language%20Models&descSize=16&descAlignY=58" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=3200&pause=800&color=2E75B6&center=true&vCenter=true&width=720&lines=The+retriever+finds+the+answer.+The+model+ignores+it.;Does+RAG+help+small+models%E2%80%94or+distract+them%3F;Six+SLMs+%C2%B7+Three+benchmarks+%C2%B7+~54%2C000+inference+calls" alt="Typing SVG" /></a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.51+-FFD21E)
![Models](https://img.shields.io/badge/SLMs-6%20(360M–3.8B)-1F4E79)
![Datasets](https://img.shields.io/badge/Benchmarks-NQ%20·%20TriviaQA%20·%20HotpotQA-2E75B6)
![Configs](https://img.shields.io/badge/Configurations-126%20main%20+%2054%20ablation-16803C)
![Status](https://img.shields.io/badge/status-research%20code-success)

</div>

---

## 🎯 The one-sentence version

> Retrieval-Augmented Generation is supposed to make models *more* factual — but for **small** language models, the retriever often surfaces the right passage and the model **still gets it wrong**. We measure exactly when, why, and by how much.

On HotpotQA the retriever puts the gold answer in front of the model **70.7%** of the time. The models answer correctly only **18.7%** of the time. That **~52-point gap** isn't a retrieval problem — it's a *utilisation* problem. This repo contains everything behind that finding.

---

## ✨ Highlights

| | |
|---|---|
| 🧪 **Full-factorial audit** | 6 SLMs × 3 datasets × 3 chunk sizes × 2 retrievers × top-*k* — **~54,000** inference calls |
| 📉 **A new metric — the Distraction Ratio** | the fraction of a model's *correct* closed-book answers that RAG *destroys* |
| 🔍 **Retrieval vs. utilisation, separated** | `recall@k` isolates "never fetched" from "fetched but unused" |
| 🧠 **Alignment beats scale** | Qwen3 and Phi-3.5-mini resist distraction due to post-training instruction alignment; a 2B model is distracted where a 0.6B one is not |
| 📊 **Reproducible** | every CSV, figure, and config in the repo; one command per stage |

---

## 🔬 The pipeline

<div align="center">
<img src="assets/Pipeline.png" width="82%" alt="RAG audit pipeline"/>
</div>

Every question is run **with** retrieval (swept across chunk size, retriever, and top-*k*) and **without** it (closed-book). The no-RAG run is the baseline that every RAG result — and the Distraction Ratio — is measured against.

---

## 📐 The Distraction Ratio ($\text{DR}_{\text{net}}$) & Instance-Level Forgetting

To quantify the harm of retrieval, we define the **Net Distraction Ratio ($\text{DR}_{\text{net}}$)**:

$$\text{DR}_{\text{net}} = \max\left(0, \frac{\text{EM}_{\text{base}} - \text{EM}_{\text{rag}}}{\text{EM}_{\text{base}}}\right) = \max\left(0, \frac{n_{10} - n_{01}}{n_{11} + n_{10}}\right), \qquad \text{EM}_{\text{base}} > 0$$

where $n_{11}, n_{10}, n_{01}, n_{00}$ represent the 2×2 question-level contingency counts between the closed-book baseline and RAG predictions ($N=300$):
- $n_{11}$: Baseline correct ($1$), RAG correct ($1$)
- $n_{10}$: Baseline correct ($1$), RAG incorrect ($0$) — *answer forgotten / model distracted*
- $n_{01}$: Baseline incorrect ($0$), RAG correct ($1$) — *retrieval helped*
- $n_{00}$: Baseline incorrect ($0$), RAG incorrect ($0$)

### Mathematical Lower Bound:
The gross **Instance-Level Distraction Ratio** is defined as:

$$\text{DR}_{\text{inst}} = \frac{n_{10}}{n_{11} + n_{10}}$$

Because $n_{01} \ge 0$, it follows directly that:

$$\text{DR}_{\text{net}} = \max\left(0, \frac{n_{10} - n_{01}}{n_{11} + n_{10}}\right) \le \frac{n_{10}}{n_{11} + n_{10}} = \text{DR}_{\text{inst}}$$

Thus, $\text{DR}_{\text{net}}$ measures net corpus-level regression below the closed-book baseline and serves as a **strict mathematical lower bound** on gross instance-level answer forgetting ($\text{DR}_{\text{net}} \le \text{DR}_{\text{inst}}$).

- `DR = 0` → retrieval did no net harm (or improved aggregate accuracy).
- `DR = 1` → retrieval destroyed *every* previously-correct answer.

Paired with `recall@k`, DR separates two failure modes:
- **high recall, low EM** → 🧠 *distraction / utilisation failure* — the gold answer was retrieved in context, but unused
- **low recall** → 🔎 *retrieval failure* — the answer was never fetched into context

---

## 📋 Experiment & Inference Call Accounting (180 Runs / 54,000 Calls)

Every single configuration is evaluated on a fixed, deterministic sample of $N=300$ questions per benchmark (`seed=42`). No inference call is duplicated or estimated:

| Experiment Segment | Configuration Breakdown | Number of Runs | Questions / Run | Total Inference Calls |
|:---|:---|:---:|:---:|:---:|
| **Closed-Book Baselines (No-RAG)** | 6 models × 3 datasets | 18 | 300 | 5,400 |
| **Main Factorial RAG Grid ($k=3$)** | 6 models × 3 datasets × 3 chunk sizes (128, 256, 512) × 2 retrievers (BM25, Dense MiniLM) | 108 | 300 | 32,400 |
| *(Main Grid Subtotal)* | *(18 baselines + 108 RAG runs = 108 RAG-vs-baseline comparisons)* | *(126)* | *(300)* | *(37,800)* |
| **Context-Depth ($k$) Ablation** | 6 models × 3 datasets × 3 top-$k$ depths ($k \in \{1, 3, 5\}$) at 256-token dense retrieval | 54 | 300 | 16,200 |
| **Grand Total** | **Full Audit Matrix** | **180** | **300** | **54,000** |

All 180 runs and all 108 paired comparisons are recorded in machine-readable CSV files produced by [`scripts/generate_audit_tables.py`](scripts/generate_audit_tables.py):
- 📄 [**`results/108_rag_vs_baseline_comparisons.csv`**](results/108_rag_vs_baseline_comparisons.csv): The 108 main RAG configurations paired against their matching closed-book baseline ($N=300$), with Wilson 95% CIs, two-proportion $z$-test $p$-values, Fisher exact $p$-values, Holm-Bonferroni adjusted $p$-values, and Benjamini-Hochberg FDR adjusted $p$-values.
- 📄 [**`results/full_180_experiment_matrix.csv`**](results/full_180_experiment_matrix.csv): The complete 180-run experiment ledger (18 baselines + 108 main RAG runs + 54 top-$k$ ablation runs = 54,000 inference calls).
- 📜 [**`scripts/generate_audit_tables.py`**](scripts/generate_audit_tables.py): Standalone reproducible script that regenerates both tables without re-running any inference.

---

## 📊 Headline results

<table>
<tr>
<td width="50%" valign="top">

**The utilisation gap** — the retriever's ceiling (dashed) vs. what models actually score.

<img src="assets/fig_gap.png" alt="Utilisation gap"/>

</td>
<td width="50%" valign="top">

**Where RAG hurts** — distraction is *localised*, not universal.

<img src="assets/fig_dr_heatmap.png" alt="Distraction Ratio heatmap"/>

</td>
</tr>
</table>

<div align="center">
<img src="assets/fig_kablation.png" width="92%" alt="Top-k ablation"/>
<br/><em>Top-<em>k</em> ablation: TriviaQA rewards more chunks, HotpotQA peaks at k=3, NQ-open saturates early.</em>
</div>

<br/>

<div align="center">

| Finding | Number |
|:--|:--:|
| RAG configs that **significantly help** (Wilson 95% CI) | **56.5%** |
| RAG configs that **significantly hurt** | **1.9%** |
| Utilisation gap on HotpotQA (recall@3 − mean EM) | **52.0 pts** |
| Largest single gap (SmolLM2-360M, TriviaQA) | **71.6 pts** |
| Distraction Ratio of Qwen3 & Phi-3.5 (all configs) | **≈ 0** |

</div>

---

## 🗂️ Repository structure

```
.
├── config.py              # models, datasets, paths, device, token loading
├── data_prep.py           # dataset download + per-question corpus build
├── chunker.py             # token-level fixed-size chunking (tiktoken cl100k_base)
├── retriever.py           # BM25 (sparse) + dense (all-MiniLM-L6-v2) retrievers
├── model_runner.py        # SLM loading + generation (Qwen3 thinking suppression)
├── metrics.py             # Exact Match, token-F1, Distraction Ratio
├── experiment.py          # main 126-configuration grid (with per-question logging)
├── k_ablation_local.py    # top-k ablation (resume/checkpoint)
├── retrieval_quality.py   # recall@k (no LLM, CPU-only)
├── significance.py        # Wilson 95% CI significance tests
├── scripts/
│   └── generate_audit_tables.py # Master reproducibility script (108 & 180 tables)
├── visualize*.py          # figures & heatmaps
├── *.ipynb                # Colab notebooks (gated-model + Phi-3.5 runs)
├── data/                  # per-question evaluation corpora (JSON)
└── results/               # output CSVs + figures
```

<details>
<summary><b>📁 Key result files (click to expand)</b></summary>

| File | Contents |
|------|----------|
| [`results/108_rag_vs_baseline_comparisons.csv`](results/108_rag_vs_baseline_comparisons.csv) | Master table: 108 RAG vs. baseline comparisons, Wilson CIs, z-test, Fisher exact, Holm & FDR adjusted $p$-values |
| [`results/full_180_experiment_matrix.csv`](results/full_180_experiment_matrix.csv) | Full 180-run ledger (18 baselines + 108 main RAG + 54 ablation = 54,000 calls) |
| `results/results.csv` | EM, F1, DR for all 126 main configurations |
| `results/k_ablation.csv` | top-*k* sweep (k ∈ {1,3,5}) |
| `results/retrieval_quality.csv` | recall@{1,3,5} for all retriever configs |
| `results/retrieval_vs_em.csv` | recall@3 vs. model EM — the utilisation gap |
| `results/significance_rag_vs_baseline.csv` | Wilson 95% CI, RAG vs. baseline |

</details>

---

## 🚀 Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Token for gated models (Gemma-2, Llama-3.2)
cp .env.example .env        # then paste your HuggingFace token

# 3. Run the audit
python data_prep.py            # build per-question corpora
python experiment.py           # main 180-config grid (--save_per_question_preds enabled by default)
python k_ablation_local.py     # top-k ablation
python retrieval_quality.py    # recall@k
python significance.py         # Wilson CI tests
python visualize.py            # figures & heatmaps
```

> 🍏 Runs on Apple Silicon (MPS) and single-GPU (Colab T4). Dense retrieval is CPU-only. Public models need no token.

---

## 📎 Citation

If this work is useful to you, please cite:

```bibtex
@inproceedings{tripathi2026distraction,
  title     = {Found but Not Used: A Full-Factorial Audit of RAG Distraction
               in Small Language Models},
  author    = {Tripathi, Swarnapravo and Kundu, Souvik and
               Sur, Sayanjib and Singh, Pawan Kumar},
  booktitle = {IEEE SILCON},
  year      = {2026}
}
```

---

## 👥 Authors

**Swarnapravo Tripathi**, **Souvik Kundu** — *Heritage Institute of Technology, Kolkata*
**Sayanjib Sur**, **Pawan Kumar Singh** — *Jadavpur University, Kolkata*

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E75B6,100:1F4E79&height=90&section=footer" width="100%"/>
</div>
