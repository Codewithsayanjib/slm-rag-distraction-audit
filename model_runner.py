"""Model loading and RAG-aware generation.

One model is held in memory at a time; loading a new model evicts the previous one.
"""
from __future__ import annotations

import gc
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, HF_TOKEN, MODELS

_model_cache:     dict[str, AutoModelForCausalLM] = {}
_tokenizer_cache: dict[str, AutoTokenizer]        = {}



def _evict_all() -> None:
    for k in list(_model_cache):
        del _model_cache[k]
        del _tokenizer_cache[k]
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    _evict_all()

    cfg      = MODELS[model_name]
    model_id = cfg["model_id"]
    dtype    = cfg["dtype"]
    print(f"\n[runner] Loading {model_name}  ({model_id})  on {DEVICE} …")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    ).to(DEVICE)
    mdl.eval()

    _model_cache[model_name]     = mdl
    _tokenizer_cache[model_name] = tok
    return mdl, tok



_SYS_NOTHINK = "Respond concisely. Do NOT use chain-of-thought or extended reasoning."

def _build_messages(model_name: str, question: str,
                    context: Optional[str]) -> list[dict]:
    if context:
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer in a short phrase (1-5 words):"
        )
    else:
        user = (
            f"Question: {question}\n"
            "Answer in a short phrase (1-5 words):"
        )

    cfg = MODELS[model_name]
    if cfg["thinking"]:
        user = "/nothink\n" + user

    return [{"role": "user", "content": user}]


def build_prompt(model_name: str, question: str,
                 context: Optional[str], tokenizer: AutoTokenizer) -> str:
    messages = _build_messages(model_name, question, context)

    cfg = MODELS[model_name]
    extra = {}
    if cfg["thinking"]:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False, **extra,
            )
        except TypeError:
            pass

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return messages[0]["content"] + "\n"



_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def generate_answer(model_name: str, question: str,
                    context: Optional[str] = None) -> str:
    mdl, tok = load_model(model_name)
    prompt   = build_prompt(model_name, question, context, tok)

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    raw     = tok.decode(new_ids, skip_special_tokens=True).strip()

    raw = _THINK_RE.sub("", raw).strip()

    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line

    return raw
