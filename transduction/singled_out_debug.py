#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug LoRA training on chat-formatted data.

Usage (QLoRA, one-batch sanity):
  python debug_lora_training.py --dataset transduction/train_dataset_singled_out.json

Usage (no quant, run tiny overfit for 300 steps):
  python debug_lora_training.py --dataset transduction/train_dataset_singled_out.json \
    --no_4bit --overfit --steps 300 --lr 5e-3 --dup 128

Test a smaller model if VRAM is tight:
  --base_model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="JSON with fields: input, output")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quant; use full precision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=300, help="Overfit steps if --overfit")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--overfit", action="store_true", help="Run tiny overfit loop on 1 sample duplicated")
    p.add_argument("--dup", type=int, default=128, help="Duplicates of single example for overfit")
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    p.add_argument("--print_every", type=int, default=20)
    return p.parse_args()

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_of(model):
    return next(model.parameters()).device

def guess_lora_targets(model) -> List[str]:
    """
    Discover common attention/MLP projection names across many architectures.
    """
    import re
    wanted = re.compile(r"(q_proj|k_proj|v_proj|o_proj|wq|wk|wv|wo|up_proj|down_proj|gate_proj)$", re.I)
    hits = set()
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if hasattr(mod, "weight") and ("linear" in cls or "quantlinear" in cls):
            short = name.split(".")[-1]
            if wanted.search(short):
                hits.add(short)
    return sorted(hits)

def print_trainables(model):
    tr = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    print(f"[trainables] tensors: {len(tr)}")
    show = min(30, len(tr))
    for n,_ in tr[:show]:
        print("   ", n)
    lora_only = [n for n,_ in tr if "lora_" in n]
    print(f"[trainables] lora-only count: {len(lora_only)}")

def lora_grad_norm(model) -> float:
    s = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora_" in n:
            s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s) if s > 0 else 0.0

def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # shift for causal LM
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != -100
    if not mask.any():
        return float("nan")

    pred = shift_logits.argmax(-1)
    return (pred[mask] == shift_labels[mask]).float().mean().item()


# -----------------------------
# Data
# -----------------------------
class ChatJsonDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "Dataset must be a list of {input, output}"
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        masked_ratios = []

        for ex in data:
            inp = ex.get("input", "").strip()
            out = ex.get("output", "").strip()
            if not inp or not out:
                continue

            messages = [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": out},
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

            full = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
            pref = tokenizer(prefix_text, truncation=True, max_length=max_len, padding=False)

            input_ids = full["input_ids"]
            attn = full["attention_mask"]
            labels = list(input_ids)

            prefix_len = len(pref["input_ids"])
            # Mask prefix tokens
            for i in range(min(prefix_len, len(labels))):
                labels[i] = -100

            # Emergency: ensure we supervise *something*
            masked = sum(1 for v in labels if v == -100)
            total = len(labels)
            if masked >= total:
                # Unmask last up to 32 tokens
                for i in range(max(0, total-32), total):
                    labels[i] = input_ids[i]
                masked = sum(1 for v in labels if v == -100)

            masked_ratios.append(masked / max(1, total))
            self.samples.append(
                {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
            )

        print(f"[data] loaded {len(self.samples)} usable samples from {path}")
        if masked_ratios:
            print(f"[data] masked ratio: avg={sum(masked_ratios)/len(masked_ratios):.3f} "
                  f"min={min(masked_ratios):.3f} max={max(masked_ratios):.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

@dataclass
class CollatorKeepLabels:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]):
        max_len = max(len(f["input_ids"]) for f in features)
        def pad(seq, val): return seq + [val] * (max_len - len(seq))
        b = {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], self.label_pad_token_id) for f in features]),
        }
        return b

# -----------------------------
# Model / LoRA
# -----------------------------
def load_model_and_tokenizer(base_model: str, use_4bit: bool, grad_ckpt: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    attn_impl = "sdpa"  # portable and fine for debugging

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    quant = None
    qcfg = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print("[warn] BitsAndBytes not available; falling back to full precision.", e)
            qcfg = None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
        quantization_config=qcfg,
        device_map="auto",
    )

    # Disable cache for training
    if hasattr(model, "config"):
        model.config.use_cache = False

    # (Optional) gradient checkpointing
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            model.gradient_checkpointing_enable()

    # Prepare for k-bit if quantized
    if qcfg is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # LoRA
    from peft import LoraConfig, get_peft_model
    targets = guess_lora_targets(model)
    print("[lora] discovered targets:", targets)
    assert targets, "No LoRA targets found! Target-module mismatch."

    lcfg = LoraConfig(
        r=32,              # keep small for debug
        lora_alpha=16,
        lora_dropout=0.0,  # 0.0 to help overfit
        target_modules=targets,
        bias="none",
        modules_to_save=[],  # keep pure LoRA for clarity
    )
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()
    print_trainables(model)

    return model, tok

# -----------------------------
# Training / Checks
# -----------------------------
def one_batch_sanity(model, batch) -> Tuple[float, float, float]:
    """
    Returns: (loss, masked_token_acc, lora_grad_norm_after_backward)
    """
    dev = device_of(model)
    batch = {k: v.to(dev) for k, v in batch.items()}
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()

    acc = masked_token_accuracy(out.logits, batch["labels"])
    gnorm = lora_grad_norm(model)
    print(f"[sanity] loss={float(loss):.4f}  token_acc(masked)={acc:.4f}  lora_grad_norm={gnorm:.6f}")
    return float(loss), float(acc), float(gnorm)

def tiny_overfit(model, loader, steps: int, lr: float, print_every: int = 20):
    dev = device_of(model)
    model.train()

    # Optimizer: try bitsandbytes 8-bit if available, else AdamW
    try:
        import bitsandbytes as bnb  # noqa: F401
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr)
        print("[optim] Using AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        print("[optim] Using torch AdamW")

    it = iter(loader)
    ema = None
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(dev) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        gnorm = lora_grad_norm(model)
        opt.step()

        acc = masked_token_accuracy(out.logits, batch["labels"])
        ema = (0.9 * ema + 0.1 * float(loss)) if ema is not None else float(loss)

        if step % print_every == 0 or step == 1:
            print(f"[train] step={step:04d}  loss={float(loss):.4f}  ema={ema:.4f}  "
                  f"acc={acc:.4f}  lora_grad_norm={gnorm:.6f}")

    return ema

# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    set_seed(args.seed)

    use_4bit = not args.no_4bit
    model, tok = load_model_and_tokenizer(args.base_model, use_4bit, args.grad_ckpt)

    # Dataset
    ds_full = ChatJsonDataset(args.dataset, tok, max_len=args.max_len)
    assert len(ds_full) > 0, "Empty dataset"

    # If overfitting: duplicate *one* example to make a tiny dataset
    if args.overfit:
        base = ds_full[0]
        dupd = [base for _ in range(args.dup)]
        class Tiny(Dataset):
            def __len__(self): return len(dupd)
            def __getitem__(self, i): return dupd[i]
        ds = Tiny()
        print(f"[overfit] using 1 sample duplicated x{args.dup}")
    else:
        ds = ds_full

    collator = CollatorKeepLabels(pad_token_id=tok.pad_token_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # --- One-batch sanity check
    print("\n=== One-batch sanity check ===")
    batch = next(iter(loader))
    loss, acc, gnorm = one_batch_sanity(model, batch)

    if math.isnan(acc):
        print("[warn] token accuracy is NaN (no unmasked labels). Check masking / chat template.")
    if gnorm == 0.0:
        print("[warn] LoRA grad-norm is 0. This usually means no trainable LoRA params got gradients "
              "(target module mismatch) OR every label is masked.")

    # --- Optional tiny overfit
    if args.overfit:
        print("\n=== Tiny overfit loop ===")
        ema = tiny_overfit(model, loader, steps=args.steps, lr=args.lr, print_every=args.print_every)

        # Re-run sanity after overfit
        print("\n=== Sanity after overfit ===")
        _ = one_batch_sanity(model, batch)

        # Quick greedy generation on the first example
        try:
            print("\n=== Greedy generation sample ===")
            # Reconstruct the user message from the dataset source (we only have tokenized tensors here),
            # so we load raw JSON to pick the very first raw item:
            with open(args.dataset, "r", encoding="utf-8") as f:
                raw = json.load(f)
            sample = raw[0]
            model.eval()
            prompt_msgs = [
                {"role": "user", "content": sample["input"]},
            ]
            prompt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

            # Ban thinking tags if present in the tokenizer vocab
            bad_words = []
            for s in ["<think>", "</think>"]:
                ids = tok(s, add_special_tokens=False).input_ids
                if ids: bad_words.append(ids)           # bad_words_ids expects List[List[int]]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,                     # greedy
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    bad_words_ids=bad_words or None,     # only if non-empty
                )

            decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(decoded)
        except Exception as e:
            print("[gen] generation failed:", e)

    print("\n=== Finished ===")

if __name__ == "__main__":
    main()
