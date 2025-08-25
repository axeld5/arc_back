"""Supervised Fine-Tuning script for ARC Transduction task — stable edition."""

import os
import platform
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer

# =========================
# Speed & safety helpers
# =========================

def pick_attn_impl(force_flash: bool = False) -> str:
    """Prefer FlashAttention2 when available, else SDPA (fast & stable)."""
    if force_flash and platform.system() == "Linux":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass
    if platform.system() == "Linux":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"

def maybe_compile(model, enable: bool, backend: str = "aot_eager", mode: str = "reduce-overhead", dynamic: bool = True):
    """
    Safe torch.compile wrapper.
    Default OFF because Qwen3-MoE + Inductor can crash with shape guards.
    If you turn it on, we use a safer backend (aot_eager) + dynamic shapes.
    """
    if not enable:
        print("[info] torch.compile disabled (safe default).")
        return model
    if not hasattr(torch, "compile"):
        print("[warn] torch.compile not available on this torch build")
        return model
    try:
        model = torch.compile(model, backend=backend, mode=mode, dynamic=dynamic, fullgraph=False)
        print(f"[info] torch.compile enabled (backend={backend}, dynamic={dynamic})")
    except Exception as e:
        print(f"[warn] torch.compile disabled due to error: {e}")
    return model

@dataclass
class DataCollatorForCausalLMWithPadding:
    """Data collator for causal language modeling with padding."""
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feat.pop("labels") for feat in features]
        batch = self.tokenizer.pad(  # type: ignore
            features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.stack([
            torch.tensor(label + [self.label_pad_token_id] * (max_len - len(label)), dtype=torch.long)
            for label in labels
        ])
        return batch  # type: ignore

def preprocess_transduction_data(example: Dict[str, Any], tokenizer: AutoTokenizer, max_len: int) -> Dict[str, Any]:
    """Preprocess transduction dataset for training."""
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
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Optional: early stop (kept from previous version; off by default)
class PlateauEarlyStop(TrainerCallback):
    def __init__(self, monitor="eval_loss", min_delta=0.0, patience=3):
        self.monitor = monitor; self.min_delta = min_delta; self.patience = patience
        self.best = None; self.bad = 0
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        m = kwargs.get("metrics", {})
        if self.monitor not in m: return
        v = m[self.monitor]
        if self.best is None or (self.best - v) > self.min_delta:
            self.best = v; self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True
                print(f"[info] Early stopping: {self.monitor} plateaued.")
        return control

def dataloader_kwargs(workers=8, pin=True, persistent=True, prefetch=6):
    return dict(
        dataloader_num_workers=workers,
        dataloader_pin_memory=pin,
        dataloader_persistent_workers=persistent,
        dataloader_prefetch_factor=prefetch,
    )

# =========================
# Main
# =========================

def main():
    """Main training function."""
    # --- Config ---
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    DATA_FILE = "transduction/train_dataset.json"
    MAX_LEN = 6000

    # Safety: keep compile OFF unless you purposely re-enable.
    USE_COMPILE = False           # <-- turn to True only if you want to try compile
    COMPILE_BACKEND = "aot_eager" # safer than inductor here
    COMPILE_DYNAMIC = True

    FORCE_FLASH = True
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 6
    USE_EARLY_STOP = False

    # Precision & TF32
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[info] bf16={'on' if use_bf16 else 'off'}")

    # --- Auth ---
    load_dotenv()
    if os.getenv("HF_TOKEN"):
        login(os.getenv("HF_TOKEN"))

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    attn_impl = pick_attn_impl(force_flash=FORCE_FLASH)
    print(f"[info] attn_implementation={attn_impl}")

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    # Gradient checkpointing (forces use_cache=False internally)
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False  # avoid warning; ensures consistency with GC

    # LoRA (attention-side)
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # or ["q_proj","k_proj","v_proj","o_proj"]
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)  # type: ignore
    model.print_trainable_parameters()  # type: ignore

    # IMPORTANT: compile AFTER PEFT wrap & GC — but it is OFF by default now
    model = maybe_compile(model, enable=USE_COMPILE, backend=COMPILE_BACKEND, dynamic=COMPILE_DYNAMIC)

    # --- Dataset ---
    print(f"Loading dataset from {DATA_FILE}...")
    raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"Loaded {len(raw_ds)} training samples")

    def preprocess_fn(example):
        return preprocess_transduction_data(example, tokenizer, MAX_LEN)

    tokenised_ds = raw_ds.map(
        preprocess_fn,
        remove_columns=raw_ds.column_names,
        num_proc=max(1, (os.cpu_count() or 4) // 2),
    )

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # --- Training args ---
    args = SFTConfig(
        output_dir="qwen3_30b_a3b_arc_transduction_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        logging_dir="transduction/tb_logs",
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        push_to_hub=True,
        hub_model_id="axel-darmouni/qwen2.5-0.5b-arc-transduction-sft",
        **dataloader_kwargs(workers=NUM_WORKERS, pin=True, persistent=True, prefetch=PREFETCH_FACTOR),
    )

    callbacks = [PlateauEarlyStop(monitor="eval_loss", min_delta=1e-3, patience=3)] if USE_EARLY_STOP else []

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=tokenised_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model("qwen3_30b_a3b_arc_transduction_sft/final")
    tokenizer.save_pretrained("qwen3_30b_a3b_arc_transduction_sft/final")

    if os.getenv("HF_TOKEN"):
        print("Pushing adapter to Hugging Face Hub...")
        try:
            trainer.push_to_hub()  # push adapter/config (not base weights)
            print("Adapter pushed successfully!")
        except Exception as e:
            print(f"[warn] Push to hub failed: {e}")
    else:
        print("No HF_TOKEN found, skipping hub push")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
