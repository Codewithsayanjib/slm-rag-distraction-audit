"""EM, token-F1, and distraction ratio."""
from __future__ import annotations

import re
import string


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = _normalize(pred)
    return float(any(p == _normalize(g) for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    p_toks = _normalize(pred).split()
    best   = 0.0
    for g in golds:
        g_toks = _normalize(g).split()
        if not p_toks or not g_toks:
            continue
        common = set(p_toks) & set(g_toks)
        if not common:
            continue
        prec = len(common) / len(p_toks)
        rec  = len(common) / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def distraction_ratio(baseline_em: float, rag_em: float) -> float:
    """Fraction of baseline EM lost after introducing RAG context.

    0 = RAG did not hurt; 1 = RAG destroyed all correct answers.
    Negative gain (RAG > baseline) clips to 0.
    """
    if baseline_em <= 1e-9:
        return 0.0
    return max(0.0, (baseline_em - rag_em) / baseline_em)
