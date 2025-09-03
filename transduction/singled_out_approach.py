"""
Singled-out training + inference pipeline for ARC Transduction.

Specs:
- Data Generation: Focus on ONE training problem, create 300 augmented versions.
  • Use ONLY one training problem (specified or first available).
  • Apply random augmentations to create multiple variants, EXCLUDING upscaling.
  • Build samples using input grids, random placeholders, and corresponding outputs.
- Train Qwen3-4B-Instruct in two stages: first SFT, then RL on the augmented single problem.
- Inference: run three attempts — AIRV, Repeat, and AIRV+Repeat — on the SAME problem used for training.

Notes:
- This file is self-contained and does not modify existing training/inference modules.
- It re-implements minimal SFT/RL loops to allow pointing at a custom dataset path.
- It implements a LoRA-aware inference helper to load base model + adapter.
- The script prints the train+test data of the chosen problem for inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import (
    list_training_problems,
    load_training_problem,
    list_evaluation_problems,
    load_evaluation_problem,
)
from transduction.data_gen import grid_to_row_strings
from augment import (
    rotate_90,
    rotate_180,
    rotate_270,
    flip_vertical,
    flip_horizontal,
)
from deaugment import apply_full_deaugmentation


# =========================
# Utility: placeholders
# =========================

# ---- LoRA target discovery (robust across Qwen variants)
def guess_lora_targets(model):
    import re
    pat = re.compile(r"(q_proj|k_proj|v_proj|o_proj|wq|wk|wv|wo|up_proj|down_proj|gate_proj)$", re.I)
    hits = set()
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if hasattr(mod, "weight") and ("linear" in cls or "quantlinear" in cls):
            short = name.split(".")[-1]
            if pat.search(short):
                hits.add(short)
    return sorted(hits)

# ---- True masked token-accuracy (uses causal shift)
import torch
def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return float("nan")
    pred = shift_logits.argmax(-1)
    return (pred[mask] == shift_labels[mask]).float().mean().item()

# ---- Constrain generation to digits / space / newline
from transformers import LogitsProcessor, LogitsProcessorList
class DigitsOnly(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tok = tokenizer
        allowed = set("0123456789 \n")
        self.allowed_ids = set()
        for tok, tid in self.tok.get_vocab().items():
            try:
                txt = self.tok.convert_tokens_to_string([tok])
            except Exception:
                txt = tok
            if txt and all(c in allowed for c in txt):
                self.allowed_ids.add(tid)
        if self.tok.eos_token_id is not None:
            self.allowed_ids.add(self.tok.eos_token_id)
    def __call__(self, input_ids, scores):
        import torch as _t
        mask = _t.full_like(scores, float("-inf"))
        idx = list(self.allowed_ids)
        mask[:, idx] = 0.0
        return scores + mask

# ---- Keep only rows that look like grids
import re as _re
def extract_grid(text: str) -> str:
    rows = [r.strip() for r in text.splitlines() if _re.fullmatch(r"[0-9 ]+", r.strip())]
    rows = [" ".join(r.split()) for r in rows if r]
    return "\n".join(rows)

# New prompt format (PROMPT_V2)
PROMPT_V2 = (
    "Solve task {task_id}\n\n"
    "INPUT:\n{input}\n"
    "OUTPUT PLACEHOLDER:\n{placeholder}\n"
    "OUTPUT:"
)

def _random_placeholder_for_grid(
    reference_grid: List[List[int]],
    all_matrices: Optional[List[List[List[int]]]] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Create a random placeholder with same shape as reference_grid.
    Strategies (no ground-truth leak): zeros, copy compatible matrix, or modified copy.
    Returns placeholder rows joined by newlines (space-separated cells).
    """
    r = rng or random
    h = len(reference_grid)
    w = len(reference_grid[0]) if h > 0 else 0

    # Collect compatible matrices (same HxW) from provided pool
    compatible: List[List[List[int]]] = []
    if all_matrices:
        for m in all_matrices:
            if m and len(m) == h and len(m[0]) == w:
                compatible.append(m)

    # Sample strategy
    strategy = r.choice(["zeros", "copy", "modified_copy"]) if compatible else "zeros"

    if strategy == "zeros":
        matrix = [[0 for _ in range(w)] for _ in range(h)]
    elif strategy == "copy":
        chosen = r.choice(compatible)
        matrix = [row[:] for row in chosen]
    else:  # modified_copy
        chosen = r.choice(compatible)
        matrix = [row[:] for row in chosen]
        num_mods = min(r.randint(1, max(1, h * w // 4)), h * w)
        positions = [(i, j) for i in range(h) for j in range(w)]
        for i, j in r.sample(positions, num_mods):
            matrix[i][j] = r.randint(0, 9)

    return "\n".join([" ".join(str(c) for c in row) for row in matrix])


# =========================
# Data generation (singled-out)
# =========================

def _format_single_prompt(input_grid: List[List[int]], placeholder_rows: str, task_id: str) -> str:
    """Format a single-input prompt with PROMPT_V2."""
    input_str = "\n".join(grid_to_row_strings(input_grid))
    return PROMPT_V2.format(task_id=task_id, input=input_str, placeholder=placeholder_rows)


def _apply_selected_augmentations_no_upscale(
    problem: Dict[str, Any],
    selected: List[str],
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Apply a sequence of augmentations to the entire problem, excluding upscaling.
    Supported ops: rotate_90/180/270, flip_vertical/horizontal, color_permutation (no upscaling).
    Returns (augmented_problem, applied_augmentations, augmentation_params) where augmentation_params
    mirrors the legacy structure expected by deaugment.apply_full_deaugmentation for color_permutation.
    """
    r = rng or random

    # Shallow copy and then mutate grids
    from copy import deepcopy

    augmented = deepcopy(problem)
    applied: List[str] = []
    aug_params: Dict[str, Any] = {}

    def _map_grids(fn):
        # Apply to all input/output grids in train/test (and ignore arc-gen by spec)
        if "train" in augmented:
            for ex in augmented["train"]:
                if "input" in ex:
                    ex["input"] = fn(ex["input"])  # type: ignore
                if "output" in ex:
                    ex["output"] = fn(ex["output"])  # type: ignore
        if "test" in augmented:
            for ex in augmented["test"]:
                if "input" in ex:
                    ex["input"] = fn(ex["input"])  # type: ignore
                if "output" in ex:
                    ex["output"] = fn(ex["output"])  # type: ignore

    for name in selected:
        if name == "color_permutation":
            colors = list(range(10))
            shuffled = colors[:]
            r.shuffle(shuffled)
            cmap = {c: shuffled[i] for i, c in enumerate(colors)}

            def _apply_cperm(grid: List[List[int]]):
                return [[cmap.get(v, v) for v in row] for row in grid]

            _map_grids(_apply_cperm)
            aug_params.setdefault("color_permutation", {})["color_map"] = cmap
            applied.append("color_permutation")
        elif name == "rotate_90":
            _map_grids(rotate_90)
            applied.append("rotate_90")
        elif name == "rotate_180":
            _map_grids(rotate_180)
            applied.append("rotate_180")
        elif name == "rotate_270":
            _map_grids(rotate_270)
            applied.append("rotate_270")
        elif name == "flip_vertical":
            _map_grids(flip_vertical)
            applied.append("flip_vertical")
        elif name == "flip_horizontal":
            _map_grids(flip_horizontal)
            applied.append("flip_horizontal")
        else:
            # Skip unsupported ops here (e.g., upscale)
            continue

    return augmented, applied, {"augmentation_params": aug_params}


def _random_augmentation_sequence_no_upscale(rng: Optional[random.Random] = None) -> List[str]:
    r = rng or random
    allowed = [
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "flip_vertical",
        "flip_horizontal",
        "color_permutation",
        # sample_permutation is ignored here because it doesn't affect test output geometry
    ]
    k = r.randint(0, 3)  # 0-3 ops
    if k == 0:
        return []
    return r.sample(allowed, k)


def generate_singled_out_dataset(
    data_dir: str = ".",
    output_file: str = "transduction/train_dataset_singled_out.json",
    problem_id: Optional[str] = None,
    num_augmentations: int = 30,
    seed: int = 42,
    apply_augmentations: bool = True,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Build dataset samples focusing on ONE problem with multiple augmentations.
    Each sample contains a single input grid as TEST INPUT, a random placeholder 
    of the same shape, and the corresponding output as target.

    - Uses only one training problem (specified or first available).
    - Excludes arc-gen.
    - Creates num_augmentations versions of the problem through augmentations.
    """
    r = random.Random(seed)
    problem_ids = list_training_problems(data_dir)
    
    if not problem_ids:
        raise ValueError("No training problems found")
    
    # Select the problem to use
    if problem_id is None:
        selected_pid = problem_ids[0]
        print(f"[info] No problem_id specified, using first available: {selected_pid}")
    else:
        if problem_id not in problem_ids:
            raise ValueError(f"Problem {problem_id} not found in training set")
        selected_pid = problem_id
    
    print(f"[info] Using problem: {selected_pid}")
    
    try:
        base_problem = load_training_problem(selected_pid, data_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load problem {selected_pid}: {e}")

    if not base_problem or not base_problem.get("train"):
        raise ValueError(f"Problem {selected_pid} has no training examples")

    # Print the problem data
    print(f"\n[info] Problem {selected_pid} details:")
    print(f"Train examples: {len(base_problem.get('train', []))}")
    print(f"Test examples: {len(base_problem.get('test', []))}")
    
    print("\nTRAIN DATA:")
    for i, ex in enumerate(base_problem.get("train", [])):
        print(f"  Example {i+1}:")
        print(f"    Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        for row in ex["input"]:
            print(f"      {' '.join(str(c) for c in row)}")
        print(f"    Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        for row in ex["output"]:
            print(f"      {' '.join(str(c) for c in row)}")
        print()
    
    print("TEST DATA:")
    for i, ex in enumerate(base_problem.get("test", [])):
        print(f"  Example {i+1}:")
        print(f"    Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        for row in ex["input"]:
            print(f"      {' '.join(str(c) for c in row)}")
        if "output" in ex:
            print(f"    Output ({len(ex['output'])}x{len(ex['output'][0])}):")
            for row in ex["output"]:
                print(f"      {' '.join(str(c) for c in row)}")
        print()

    samples: List[Dict[str, str]] = []

    # Generate augmented versions
    for aug_idx in range(num_augmentations):
        problem = json.loads(json.dumps(base_problem))  # Deep copy
        
        # Apply augmentations (exclude upscaling)
        if apply_augmentations:
            selected = _random_augmentation_sequence_no_upscale(r)
            if selected:
                problem, _, _ = _apply_selected_augmentations_no_upscale(problem, selected, r)

        # Build matrix pool for placeholders (inputs/outputs from train + test inputs only)
        matrix_pool: List[List[List[int]]] = []
        for ex in problem.get("train", []):
            if "input" in ex:
                matrix_pool.append(ex["input"])  # type: ignore
            if "output" in ex:
                matrix_pool.append(ex["output"])  # type: ignore
        for ex in problem.get("test", []):
            if "input" in ex:
                matrix_pool.append(ex["input"])  # type: ignore

        # Create one sample per available train pair for this augmentation
        for ex in problem.get("train", []):
            input_grid = ex.get("input")
            output_grid = ex.get("output")
            if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                continue

            placeholder = _random_placeholder_for_grid(input_grid, matrix_pool, r)
            prompt = _format_single_prompt(input_grid, placeholder, task_id=selected_pid)
            target_output_str = "\n".join(grid_to_row_strings(output_grid))
            samples.append({"input": prompt, "output": target_output_str})

        if (aug_idx + 1) % 50 == 0:
            print(f"[info] Generated {aug_idx+1}/{num_augmentations} augmented versions")

    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[info] Saved {len(samples)} samples to {output_file}")
    print(f"[info] Total samples: {len(samples)} (from {num_augmentations} augmentations × {len(base_problem['train'])} train examples)")
    
    # Calculate token statistics using tiktoken
    try:
        import tiktoken
        
        # Use cl100k_base encoding (GPT-4 tokenizer) as a standard reference
        encoding = tiktoken.get_encoding("cl100k_base")
        
        token_counts = []
        for sample in samples:
            # Count tokens for the full sample (input + output)
            full_text = sample["input"] + "\n" + sample["output"]
            token_count = len(encoding.encode(full_text))
            token_counts.append(token_count)
        
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            print(f"[info] Token statistics (using tiktoken cl100k_base):")
            print(f"[info]   Average tokens per sample: {avg_tokens:.1f}")
            print(f"[info]   Min tokens: {min_tokens}")
            print(f"[info]   Max tokens: {max_tokens}")
        else:
            print("[info] No samples to analyze for token statistics")
            
    except ImportError:
        print("[warning] tiktoken not available, skipping token statistics")
    except Exception as e:
        print(f"[warning] Error calculating token statistics: {e}")
    return samples, selected_pid


# =========================
# Training: SFT and RL
# =========================

# drop-in, works with pre-tokenized examples that already have input_ids/attention_mask/labels
from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class DataCollatorKeepLabels:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        def pad(seq, val): return seq + [val] * (max_len - len(seq))
        batch = {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features]),
            "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.tensor([pad(f["labels"], self.label_pad_token_id) for f in features]),
        }
        return batch



def run_sft(
    dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_sft",
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    learning_rate: float = 5e-4,
    num_train_epochs: int = 20,
    grad_accum: int = 8,
    batch_size: int = 1,
    use_compile: bool = False,
):
    """Run minimal SFT on the singled-out dataset with LoRA."""
    import platform
    import torch
    from datasets import load_dataset
    from dotenv import load_dotenv
    from huggingface_hub import login
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    def pick_attn_impl() -> str:
        if platform.system() == "Linux":
            try:
                import importlib
                importlib.import_module("flash_attn")
                return "flash_attention_2"
            except Exception:
                return "sdpa"
        return "sdpa"

    def preprocess(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        full_tokens = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
        prefix_tokens = tokenizer(prefix_text, truncation=True, max_length=max_len, padding=False)

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = list(input_ids)

        prefix_len = len(prefix_tokens["input_ids"])
        # mask prefix
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

        # emergency guard: ensure supervision exists
        if all(v == -100 for v in labels):
            for i in range(max(0, len(labels) - 32), len(labels)):
                labels[i] = input_ids[i]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Auth
    load_dotenv()
    if os.getenv("HF_TOKEN"):
        try:
            login(os.getenv("HF_TOKEN"))
        except Exception:
            pass

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = pick_attn_impl()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=compute_dtype,
        trust_remote_code=True,
        quantization_config=quant,
        attn_implementation=attn_impl,
    )

    # Important order: disable cache -> (optionally) grad ckpt -> prepare k-bit -> attach LoRA
    if hasattr(model, "config"): model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        model.gradient_checkpointing_enable()

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # Auto-discover targets (includes MLP)
    targets = guess_lora_targets(model)
    print("[lora] discovered targets:", targets)
    assert targets, "No LoRA targets found!"

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.0,      # 0.0 helps tiny-data overfit; bump to 0.05 later
        target_modules=targets,
        bias="none",
        modules_to_save=[],    # keep pure LoRA to avoid confusion
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()

    try:
        model.lm_head.to(dtype=compute_dtype, device=next(model.parameters()).device)
    except Exception:
        pass

    # Verify trainables
    trainables = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    print(f"[lora] trainable tensors: {len(trainables)}")

    # Dataset
    raw = load_dataset("json", data_files=dataset_path, split="train")
    tokenised = raw.map(lambda ex: preprocess(ex, tokenizer, 4096), remove_columns=raw.column_names, num_proc=max(1, (os.cpu_count() or 4) // 2))

    # Trainer
    args = SFTConfig(
        output_dir=output_dir,
        packing=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        max_grad_norm=None,
    )

    from trl import SFTTrainer

    collator = DataCollatorKeepLabels(pad_token_id=tokenizer.pad_token_id)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=tokenised,
        data_collator=collator,  # <- critical
    )
    b = next(iter(trainer.get_train_dataloader()))
    lab = b["labels"]
    num_unmasked = int((lab != -100).sum().item())

    # autocast context for manual forward (Trainer handles this automatically during .train())
    if torch.cuda.is_available():
        from torch.cuda.amp import autocast
        amp_ctx = autocast(dtype=compute_dtype)
    else:
        amp_ctx = contextlib.nullcontext()

    with torch.no_grad(), amp_ctx:
        out = model(
            input_ids=b["input_ids"].to(model.device),
            attention_mask=b["attention_mask"].to(model.device),
            labels=b["labels"].to(model.device),
        )

    from_here_labels = b["labels"].to(model.device)  # for metric dtype alignment
    acc = masked_token_accuracy(out.logits, from_here_labels)
    print("[collate] labels shape:", tuple(lab.shape), "unmasked tokens:", num_unmasked,
        "token_acc(masked):", f"{acc:.4f}")
    assert num_unmasked > 0, "All labels masked after collation!"
        
    print("[sft] Starting training...")
    trainer.train()
    print("[sft] Saving final adapter...")
    trainer.save_model(os.path.join(output_dir, "final"))
    try:
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    except Exception:
        pass
    
    # Simple inference evaluation after SFT
    print("[sft] Running simple inference evaluation...")
    try:
        sample_data = raw[0]
        messages = [
            {"role": "user", "content": sample_data["input"]},
        ]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        eval_inputs = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=2048)
        eval_inputs = {k: v.to(model.device) for k, v in eval_inputs.items()}

        processors = LogitsProcessorList([DigitsOnly(tokenizer)])
        with torch.no_grad():
            eval_outputs = model.generate(
                **eval_inputs,
                max_new_tokens=2048,
                do_sample=False,  # greedy for structure
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(eval_outputs[0][eval_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = extract_grid(decoded)

        print("[sft] Sample evaluation:")
        print(f"[sft] Input:\n{sample_data['input'][:300]}...")
        print(f"[sft] Expected:\n{sample_data['output'][:300]}...")
        print(f"[sft] Generated (cleaned):\n{cleaned[:300]}...")
        print("[sft] SFT evaluation complete.")
    except Exception as e:
        print(f"[sft] Evaluation failed: {e}")
    
    return os.path.join(output_dir, "final")


def run_rl(
    base_model: str,
    lora_path: str,
    dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_rl",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    grad_accum: int = 4,
    num_generations: int = 4,
):
    """Run minimal GRPO on top of SFT LoRA using the same dataset."""
    import platform
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer
    from transduction.training.reward_fn import reward_function

    # Tokenizer (prefer LoRA path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if platform.system() == "Linux" else "sdpa"
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.enable_input_require_grads()

    # Build RL dataset format
    raw_ds = load_dataset("json", data_files=dataset_path, split="train")

    def to_rl(ex):
        messages = [{"role":"user","content": ex["input"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        expected_grid = []
        if ex.get("output"):
            rows = ex["output"].strip().split("\n")
            for row in rows:
                if row.strip():
                    expected_grid.append([int(x) for x in row.strip().split()])
        return {"prompt": prompt_text, "expected_output": expected_grid}

    ds = raw_ds.map(to_rl, remove_columns=raw_ds.column_names, num_proc=4)

    # Build mapping for reward lookup
    p2y = {rec["prompt"]: rec["expected_output"] for rec in ds}

    # Global counter for logging
    generation_counter = [0]  # Use list to allow modification in nested function
    
    def contextual_reward(completions: List[str], prompts: List[str], **kwargs: Any) -> List[float]:
        expected = [p2y.get(p, []) for p in prompts]
        rewards = [float(r) for r in reward_function(completions, expected)]
        
        # Print outputs every 5 logging steps (logging_steps=10, so every 50 generations)
        generation_counter[0] += 1
        if generation_counter[0] % 50 == 0:
            print(f"\n[rl] Generation batch #{generation_counter[0]}:")
            for i, (completion, reward) in enumerate(zip(completions[:3], rewards[:3])):  # Show first 3
                print(f"[rl] Sample {i+1} (reward={reward:.3f}):")
                print(f"[rl] Completion: {completion[:150]}...")
            print("[rl] ---")
        
        return rewards

    cfg = GRPOConfig(
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        beta=0.04,
        epsilon=3e-4,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.20,
        num_generations=num_generations,
        max_prompt_length=4096,
        max_completion_length=2048,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[contextual_reward], args=cfg, train_dataset=ds)
    print("[rl] Starting training...")
    trainer.train()
    print("[rl] Saving final adapter...")
    trainer.save_model(os.path.join(output_dir, "final"))
    try:
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    except Exception:
        pass
    return os.path.join(output_dir, "final")


# =========================
# LoRA-aware inference helper
# =========================

class LoRAARCTransductionInference:
    def __init__(self, base_model: str, lora_path: Optional[str] = None, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        from peft import PeftModel

        self.device = self._get_device(device)
        # Tokenizer: try LoRA path first for added tokens
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(lora_path or base_model, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto" if self.device != "cpu" else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        # Resize embeddings if tokenizer changed
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.vocab_size = len(self.tokenizer)
            self.model.config.bos_token_id = self.tokenizer.bos_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _get_device(self, device: str) -> str:
        import torch
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _format_prompt(self, problem_data: Dict[str, Any], train_sample_count: int = 3, test_example_idx: int = 0, task_id: str = "task") -> str:
        test_examples = problem_data.get("test", [])
        if not test_examples:
            raise ValueError("No test examples available")
        test_example = test_examples[test_example_idx % len(test_examples)]
        input_str = "\n".join(grid_to_row_strings(test_example["input"]))
        rows = len(test_example["input"]) if isinstance(test_example.get("input"), list) else 0
        cols = len(test_example["input"][0]) if rows > 0 else 0
        placeholder = "\n".join([" ".join(["0"] * cols) for _ in range(rows)])
        return PROMPT_V2.format(task_id=task_id, input=input_str, placeholder=placeholder)

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        from transformers import LogitsProcessorList
        processors = LogitsProcessorList([DigitsOnly(self.tokenizer)])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                logits_processor=processors,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return extract_grid(raw)

    @staticmethod
    def parse_grid(response: str) -> Optional[List[List[int]]]:
        import re
        s = response.strip()
        if "\n" in s:
            m = re.search(r"[0-9\n\s]+", s)
            if m:
                grid_str = m.group()
                try:
                    rows = grid_str.split("\n")
                    grid: List[List[int]] = []
                    for row in rows:
                        if row.strip():
                            parts = row.strip().split()
                            if len(parts) > 1:
                                grid_row = [int(p) for p in parts if p.isdigit() or (p and p[0] == '-' and p[1:].isdigit())]
                            else:
                                grid_row = [int(ch) for ch in row if ch.isdigit()]
                            if grid_row:
                                grid.append(grid_row)
                    if grid:
                        return grid
                except Exception:
                    pass
        return None

    @staticmethod
    def evaluate(pred: Optional[List[List[int]]], gt: List[List[int]]) -> bool:
        return pred == gt if pred is not None else False

    def infer_single_problem(self, problem_data: Dict[str, Any], train_sample_count: int = 3, test_example_idx: int = 0, verbose: bool = False, task_id: Optional[str] = None) -> Dict[str, Any]:
        prompt = self._format_prompt(problem_data, train_sample_count, test_example_idx, task_id or "task")
        resp = self.generate(prompt)
        pred = self.parse_grid(resp)
        gt = problem_data["test"][test_example_idx % len(problem_data["test"])]["output"]
        res = {
            "prompt": prompt,
            "response": resp,
            "predicted_grid": pred,
            "ground_truth": gt,
            "is_correct": self.evaluate(pred, gt),
            "train_sample_count": train_sample_count,
            "test_example_idx": test_example_idx,
        }
        if verbose:
            print(f"[infer] correct={res['is_correct']}")
        return res


# =========================
# Inference strategies (no-upscale AIRV, Repeat, AIRV+Repeat)
# =========================

def _create_augmented_versions_no_upscale(
    problem_data: Dict[str, Any],
    num_versions: int = 8,
    include_original: bool = True,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Dict[str, Any], List[str], Dict[str, Any]]]:
    r = rng or random
    versions: List[Tuple[Dict[str, Any], List[str], Dict[str, Any]]] = []
    if include_original:
        versions.append((json.loads(json.dumps(problem_data)), [], {}))
    for _ in range(num_versions):
        seq = _random_augmentation_sequence_no_upscale(r)
        aug_prob, applied, meta = _apply_selected_augmentations_no_upscale(problem_data, seq, r)
        versions.append((aug_prob, applied, meta))
    return versions


def _revert_predicted_grid(
    predicted_grid: Optional[List[List[int]]],
    augmentation_list: List[str],
    metadata: Dict[str, Any],
) -> Optional[List[List[int]]]:
    if predicted_grid is None or not augmentation_list:
        return predicted_grid
    try:
        dummy_problem = {"test": [{"output": predicted_grid}]}
        reverted = apply_full_deaugmentation(dummy_problem, augmentation_list, metadata)
        return reverted["test"][0]["output"]
    except Exception:
        return None


def _vote_on_grids(valid: List[List[List[int]]]) -> Tuple[Optional[List[List[int]]], Dict[str, int]]:
    if not valid:
        return None, {}
    import json as _json
    counts: Dict[str, int] = {}
    for g in valid:
        s = _json.dumps(g)
        counts[s] = counts.get(s, 0) + 1
    winner = max(counts, key=counts.get)
    return _json.loads(winner), counts


def run_airv_no_upscale(
    inference: LoRAARCTransductionInference,
    problem_data: Dict[str, Any],
    train_sample_count: int = 3,
    test_example_idx: int = 0,
    num_versions: int = 8,
    include_original: bool = True,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    versions = _create_augmented_versions_no_upscale(problem_data, num_versions, include_original)
    results = []
    for i, (prob_v, augs, meta) in enumerate(versions):
        try:
            r = inference.infer_single_problem(prob_v, train_sample_count, test_example_idx, verbose=False, task_id=task_id)
            results.append({"version_idx": i, "augmentations": augs, "metadata": meta, "raw_prediction": r["predicted_grid"]})
        except Exception as e:
            print(f"[warn] AIRV inference failed on version {i}: {e}")

    reverted: List[List[List[int]]] = []
    reversion_info: List[Dict[str, Any]] = []
    for rec in results:
        if not rec["augmentations"]:
            if rec["raw_prediction"] is not None:
                reverted.append(rec["raw_prediction"])
                reversion_info.append({"version_idx": rec["version_idx"], "reversion_success": True})
            continue
        rev = _revert_predicted_grid(rec["raw_prediction"], rec["augmentations"], rec["metadata"])
        if rev is not None:
            reverted.append(rev)
            reversion_info.append({"version_idx": rec["version_idx"], "reversion_success": True})
        else:
            reversion_info.append({"version_idx": rec["version_idx"], "reversion_success": False})

    pred, vote_counts = _vote_on_grids(reverted)
    gt = problem_data["test"][test_example_idx % len(problem_data["test"])]["output"]
    is_correct = LoRAARCTransductionInference.evaluate(pred, gt)
    return {
        "predicted_grid": pred,
        "ground_truth": gt,
        "is_correct": is_correct,
        "num_versions": len(versions),
        "valid_outputs": len(reverted),
        "vote_counts": vote_counts,
        "reversion_info": reversion_info,
    }


def run_repeat(
    inference: LoRAARCTransductionInference,
    problem_data: Dict[str, Any],
    train_sample_count: int = 3,
    test_example_idx: int = 0,
    num_passes: int = 2,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    # First pass
    first = inference.infer_single_problem(problem_data, train_sample_count, test_example_idx, verbose=False, task_id=task_id)
    test_ex = problem_data["test"][test_example_idx % len(problem_data["test"])].copy()

    # Build placeholder from first prediction (or zeros if None)
    def _grid_to_placeholder(g: Optional[List[List[int]]], ref: List[List[int]]) -> str:
        if g is None:
            rows, cols = len(ref), (len(ref[0]) if ref else 0)
            return "\n".join([" ".join(["0"] * cols) for _ in range(rows)])
        return "\n".join([" ".join(str(c) for c in row) for row in g])

    placeholder = _grid_to_placeholder(first["predicted_grid"], test_ex["input"])

    # Build PROMPT_V2
    test_input_str = "\n".join(grid_to_row_strings(test_ex["input"]))
    prompt = PROMPT_V2.format(task_id=(task_id or "task"), input=test_input_str, placeholder=placeholder)

    resp = inference.generate(prompt)
    pred = LoRAARCTransductionInference.parse_grid(resp)
    gt = test_ex["output"]
    results = [first]
    results.append({
        "prompt": prompt,
        "response": resp,
        "predicted_grid": pred,
        "ground_truth": gt,
        "is_correct": LoRAARCTransductionInference.evaluate(pred, gt),
    })
    final = results[-1]
    final["all_pass_results"] = [
        {"pass_idx": 1, "predicted_grid": results[0].get("predicted_grid"), "is_correct": results[0].get("is_correct")},
        {"pass_idx": 2, "predicted_grid": final.get("predicted_grid"), "is_correct": final.get("is_correct")},
    ]
    final["pass_count"] = min(num_passes, 2)
    final["inference_method"] = "repeat_placeholder"
    return final


def run_airv_plus_repeat(
    inference: LoRAARCTransductionInference,
    problem_data: Dict[str, Any],
    train_sample_count: int = 3,
    test_example_idx: int = 0,
    num_versions: int = 8,
    include_original: bool = True,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    # AIRV first
    airv_res = run_airv_no_upscale(inference, problem_data, train_sample_count, test_example_idx, num_versions, include_original, task_id=task_id)
    test_ex = problem_data["test"][test_example_idx % len(problem_data["test"])].copy()

    # Repeat pass using AIRV-voted prediction as placeholder
    def _grid_to_placeholder(g: Optional[List[List[int]]], ref: List[List[int]]) -> str:
        if g is None:
            rows, cols = len(ref), (len(ref[0]) if ref else 0)
            return "\n".join([" ".join(["0"] * cols) for _ in range(rows)])
        return "\n".join([" ".join(str(c) for c in row) for row in g])

    placeholder = _grid_to_placeholder(airv_res["predicted_grid"], test_ex["input"])
    test_input_str = "\n".join(grid_to_row_strings(test_ex["input"]))
    prompt = PROMPT_V2.format(task_id=(task_id or "task"), input=test_input_str, placeholder=placeholder)
    resp = inference.generate(prompt)
    pred = LoRAARCTransductionInference.parse_grid(resp)
    gt = test_ex["output"]
    return {
        "airv_pred": airv_res["predicted_grid"],
        "final_pred": pred,
        "ground_truth": gt,
        "is_correct": LoRAARCTransductionInference.evaluate(pred, gt),
        "airv_meta": {k: airv_res[k] for k in ("num_versions", "valid_outputs", "vote_counts")},
    }


# =========================
# CLI Orchestrator
# =========================

def main():
    parser = argparse.ArgumentParser(description="Singled-out ARC Transduction pipeline")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--dataset_out", type=str, default="transduction/train_dataset_singled_out.json")
    parser.add_argument("--problem_id", type=str, default=None, help="Specific problem ID to use for training. If None, uses first available.")
    parser.add_argument("--num_augmentations", type=int, default=3, help="Number of augmented versions to generate from the single problem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_rl", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--sft_out", type=str, default="qwen3_4b_singled_out_sft")
    parser.add_argument("--rl_out", type=str, default="qwen3_4b_singled_out_rl")
    parser.add_argument("--airv_versions", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    dataset_path = args.dataset_out
    training_problem_id = None
    
    if not args.skip_gen:
        _, training_problem_id = generate_singled_out_dataset(
            data_dir=args.data_dir,
            output_file=dataset_path,
            problem_id=args.problem_id,
            num_augmentations=args.num_augmentations,
            seed=args.seed,
            apply_augmentations=not args.no_augment,
        )
    else:
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Remove --skip_gen or fix path.")
        # If we're skipping generation, we need to determine which problem to use for inference
        if args.problem_id:
            training_problem_id = args.problem_id
        else:
            # Use first available training problem
            import random
            problem_ids = list_training_problems(args.data_dir)
            if not problem_ids:
                raise RuntimeError("No training problems available")
            training_problem_id = random.choice(problem_ids)
            print(f"[info] Using first available problem for inference: {training_problem_id}")

    sft_final = os.path.join(args.sft_out, "final")
    if not args.skip_sft:
        sft_final = run_sft(dataset_path=dataset_path, output_dir=args.sft_out, base_model=args.base_model)
    else:
        if not Path(sft_final).exists():
            raise FileNotFoundError(f"SFT adapter not found at {sft_final}. Remove --skip_sft or fix path.")

    rl_final = os.path.join(args.rl_out, "final")
    if not args.skip_rl:
        rl_final = run_rl(base_model=args.base_model, lora_path=sft_final, dataset_path=dataset_path, output_dir=args.rl_out)
    else:
        if not Path(rl_final).exists():
            print(f"[warn] RL adapter not found at {rl_final}. Falling back to SFT adapter for inference.")
            rl_final = sft_final

    if args.skip_infer:
        return

    # Use the same problem that was used for training
    pid = training_problem_id
    print(f"[infer] Evaluating on the same problem used for training: {pid}")
    problem = load_training_problem(pid, args.data_dir)
    
    # Use problem id as task_id
    task_id = pid

    # LoRA-aware inference using RL (or SFT) adapter
    inf = LoRAARCTransductionInference(base_model=args.base_model, lora_path=rl_final, device=args.device)

    # Attempt 1: AIRV (no upscaling)
    airv = run_airv_no_upscale(inf, problem, train_sample_count=3, test_example_idx=0, num_versions=args.airv_versions, include_original=True, task_id=task_id)
    print(f"[AIRV] correct={airv['is_correct']} votes={airv['vote_counts']}")

    # Attempt 2: Repeat
    repeat = run_repeat(inf, problem, train_sample_count=3, test_example_idx=0, num_passes=2, task_id=task_id)
    print(f"[Repeat] correct={repeat['is_correct']}")

    # Attempt 3: AIRV + Repeat
    combo = run_airv_plus_repeat(inf, problem, train_sample_count=3, test_example_idx=0, num_versions=args.airv_versions, include_original=True, task_id=task_id)
    print(f"[AIRV+Repeat] correct={combo['is_correct']}")

    # Save a small summary next to dataset
    summary = {
        "training_problem_id": pid,
        "note": "Evaluated on the same problem used for training (with augmentations)",
        "num_augmentations_used": args.num_augmentations,
        "airv": {k: airv[k] for k in ("is_correct", "vote_counts")},
        "repeat": {"is_correct": repeat.get("is_correct")},
        "airv_plus_repeat": {"is_correct": combo.get("is_correct")},
    }
    with open("transduction/singled_out_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[infer] Results saved to transduction/singled_out_results.json")
    print(f"[infer] Training and evaluation completed on problem: {pid}")


if __name__ == "__main__":
    main()


