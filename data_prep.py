"""Download Q/A pairs + source documents for NQ-open, TriviaQA, and HotpotQA.

Documents:
  nq        – Wikipedia articles fetched via the `wikipedia` package (cached to cache/)
  triviaqa  – Wikipedia articles bundled in the dataset's entity_pages field (no API calls)
  hotpotqa  – Multi-paragraph context bundled in the dataset (supporting + distractor paras)

Run once per dataset: python data_prep.py [nq|triviaqa|hotpotqa]
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from config import CACHE_DIR, DATA_DIR, NUM_SAMPLES


# ── helpers ─────────────────────────────────────────────────────────────────

def _cache_key(question: str) -> str:
    return re.sub(r"[^\w]", "_", question[:60]).strip("_")


def _fetch_wikipedia(question: str, answers: list[str]) -> str | None:
    """Try multiple queries; return Wikipedia article text or None."""
    import wikipedia as _wp
    queries = [question] + [a for a in answers[:2]]
    for q in queries:
        try:
            hits = _wp.search(q, results=4)
            for title in hits:
                try:
                    page = _wp.page(title, auto_suggest=False)
                    if len(page.content.split()) >= 250:
                        return page.content
                except (_wp.DisambiguationError, _wp.PageError):
                    continue
        except Exception:
            continue
    return None


# ── per-dataset prep functions ───────────────────────────────────────────────

def prepare_nq(n: int = NUM_SAMPLES, force_refresh: bool = False) -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    out = DATA_DIR / "nq_corpus.json"
    if out.exists() and not force_refresh:
        print("[data_prep] Loading cached NQ corpus …")
        return json.loads(out.read_text())

    import wikipedia as _wp
    print("[data_prep] Loading NQ-open validation split …")
    nq = load_dataset("google-research-datasets/nq_open", split="validation")
    _wp.set_lang("en")

    samples: list[dict] = []
    nq_idx = 0
    with tqdm(total=n, desc="NQ: fetching Wikipedia articles") as pbar:
        while len(samples) < n and nq_idx < len(nq):
            item     = nq[nq_idx]; nq_idx += 1
            question = item["question"]
            answers  = item["answer"]
            if not answers:
                continue

            ck   = _cache_key(question)
            path = CACHE_DIR / f"nq_{ck}.txt"
            if path.exists():
                article = path.read_text(encoding="utf-8")
            else:
                article = _fetch_wikipedia(question, answers)
                if article is None:
                    continue
                path.write_text(article, encoding="utf-8")
                time.sleep(0.25)

            if len(article.split()) < 250:
                continue

            samples.append({
                "id": len(samples), "dataset": "nq",
                "question": question, "answers": answers, "document": article,
            })
            pbar.update(1)

    print(f"[data_prep] NQ: {len(samples)} samples")
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    return samples


def prepare_triviaqa(n: int = NUM_SAMPLES, force_refresh: bool = False) -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)

    out = DATA_DIR / "triviaqa_corpus.json"
    if out.exists() and not force_refresh:
        print("[data_prep] Loading cached TriviaQA corpus …")
        return json.loads(out.read_text())

    print("[data_prep] Loading TriviaQA rc validation split …")
    # 'rc' config bundles Wikipedia entity_pages — no external API calls needed
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")

    samples: list[dict] = []
    with tqdm(total=n, desc="TriviaQA: processing") as pbar:
        for item in ds:
            if len(samples) >= n:
                break

            question = item["question"]
            # normalized_aliases gives cleaned answer variants (best for EM)
            answers  = item["answer"].get("normalized_aliases") or \
                       [item["answer"]["normalized_value"]]
            answers  = [a for a in answers if a.strip()]
            if not answers:
                continue

            # entity_pages.wiki_context is a list of Wikipedia article strings
            wiki_contexts = item.get("entity_pages", {}).get("wiki_context", [])
            wiki_contexts = [w for w in wiki_contexts if len(w.split()) >= 50]
            if not wiki_contexts:
                continue

            # Join all retrieved entity articles as the document
            document = "\n\n".join(wiki_contexts)
            if len(document.split()) < 250:
                continue

            samples.append({
                "id": len(samples), "dataset": "triviaqa",
                "question": question, "answers": answers, "document": document,
            })
            pbar.update(1)

    print(f"[data_prep] TriviaQA: {len(samples)} samples")
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    return samples


def prepare_hotpotqa(n: int = NUM_SAMPLES, force_refresh: bool = False) -> list[dict]:
    DATA_DIR.mkdir(exist_ok=True)

    out = DATA_DIR / "hotpotqa_corpus.json"
    if out.exists() and not force_refresh:
        print("[data_prep] Loading cached HotpotQA corpus …")
        return json.loads(out.read_text())

    print("[data_prep] Loading HotpotQA distractor validation split …")
    # 'distractor' config: each item has 2 gold + 8 distractor paragraphs
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    samples: list[dict] = []
    with tqdm(total=n, desc="HotpotQA: processing") as pbar:
        for item in ds:
            if len(samples) >= n:
                break

            question = item["question"]
            answer   = item["answer"].strip()
            if not answer:
                continue

            # context: dict with 'title' (list) and 'sentences' (list of lists)
            titles    = item["context"]["title"]
            sent_lists = item["context"]["sentences"]

            paragraphs = []
            for title, sents in zip(titles, sent_lists):
                para = title + ". " + " ".join(sents)
                paragraphs.append(para)

            document = "\n\n".join(paragraphs)
            if len(document.split()) < 100:   # HotpotQA docs are shorter by design
                continue

            samples.append({
                "id": len(samples), "dataset": "hotpotqa",
                "question": question, "answers": [answer], "document": document,
            })
            pbar.update(1)

    print(f"[data_prep] HotpotQA: {len(samples)} samples")
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    return samples


# ── unified dispatcher ───────────────────────────────────────────────────────

def prepare_dataset(name: str, n: int = NUM_SAMPLES,
                    force_refresh: bool = False) -> list[dict]:
    """Return up to `n` samples for the named dataset."""
    dispatch = {
        "nq":       prepare_nq,
        "triviaqa": prepare_triviaqa,
        "hotpotqa": prepare_hotpotqa,
    }
    if name not in dispatch:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(dispatch)}")
    return dispatch[name](n=n, force_refresh=force_refresh)


# kept for backward-compat with old experiment.py imports
def prepare_data(force_refresh: bool = False) -> list[dict]:
    return prepare_nq(force_refresh=force_refresh)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "nq"
    data = prepare_dataset(name)
    s    = data[0]
    print(f"\nDataset : {name}")
    print(f"Sample 0:")
    print(f"  Q   : {s['question']}")
    print(f"  A   : {s['answers'][:3]}")
    print(f"  Doc : {len(s['document'].split())} words")
