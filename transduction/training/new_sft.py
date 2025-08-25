"""Supervised Fine-Tuning script for ARC Transduction task — fast edition (no QLoRA changes)."""

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

from trl import (
    SFTConfig,
    SFTTrainer,
)

# ---------------------------
# Speed helpers
# ---------------------------

def pick_attn_impl(force_flash: bool = False) -> str:
    """Prefer FlashAttention2 when available, else SDPA (fast & stable)."""
    if force_flash:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass
    # On Linux, try flash-attn automatically if present
    if platform.system() == "Linux":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    # Non-Linux: SDPA is usually faster/safer than eager
    return "sdpa"

def maybe_compile(model, enable: bool, mode: str = "reduce-overhead", fullgraph: bool = False):
    """Safe torch.compile wrapper."""
    if not enable:
        return model
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=mode, fullgraph=fullgraph)
            print("[info] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}")
    else:
        print("[warn] torch.compile not available on this torch build")
    return model

@dataclass
class BatchSizer:
    """Dynamic microbatching by tokens per device."""
    max_tokens_per_device: int          # e.g., 131072
    seq_len: int                        # max_seq_length
    starting_microbatch: int = 1
    world_size: Optional[int] = None

    def suggest(self, gradient_accumulation_steps: int) -> Tuple[int, int]:
        ws = self.world_size or int(os.environ.get("WORLD_SIZE", "1"))
        per_dev = self.starting_microbatch
        while True:
            tokens_per_dev = per_dev * self.seq_len
            if tokens_per_dev > self.max_tokens_per_device:
                per_dev = max(1, per_dev // 2)
                break
            if tokens_per_dev * 2 > self.max_tokens_per_device:
                break
            per_dev *= 2
        # keep global tokens per step ~constant
        new_ga = max(1, gradient_accumulation_steps * max(1, self.starting_microbatch) // max(1, per_dev))
        print(f"[info] BatchSizer -> per_device_train_batch_size={per_dev}, grad_accum={new_ga}")
        return per_dev, new_ga

def dataloader_kwargs(
    workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
):
    return dict(
        dataloader_num_workers=workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=persistent_workers,
        dataloader_prefetch_factor=prefetch_factor,
    )

class PlateauEarlyStop(TrainerCallback):
    """Stop if eval_loss doesn't improve by min_delta for 'patience' evals (only triggers if you pass eval_dataset)."""
    def __init__(self, monitor="eval_loss", min_delta=0.0, patience=3):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.bad_count = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        metrics = kwargs.get("metrics", {})
        if self.monitor not in metrics:
            return
        val = metrics[self.monitor]
        if self.best is None or (self.best - val) > self.min_delta:
            self.best = val
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count >= self.patience:
                control.should_training_stop = True
                print(f"[info] Early stopping: {self.monitor} plateaued.")
        return control

# ---------------------------
# Your original collator & preprocessing
# ---------------------------

@dataclass
class DataCollatorForCausalLMWithPadding:
    """Data collator for causal language modeling with padding."""
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8  # keep your 8-byte alignment
    label_pad_token_id: int = -100      # ignored by the loss

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feat.pop("labels") for feat in features]
        batch = self.tokenizer.pad(  # type: ignore
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.stack([
            torch.tensor(label + [self.label_pad_token_id] * (max_len - len(label)), dtype=torch.long)
            for label in labels
        ])
        return batch  # type: ignore


def preprocess_transduction_data(example: Dict[str, Any], tokenizer: AutoTokenizer, max_len: int) -> Dict[str, Any]:
    """
    Preprocess transduction dataset for training.

    Args:
        example: Dict with 'input' (prompt) and 'output' (target)
        tokenizer: Tokenizer to use
        max_len: Maximum sequence length
    """
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_tokens = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
    prefix_tokens = tokenizer(prefix_text, truncation=True, max_length=max_len, padding=False)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = list(input_ids)
    prefix_len = len(prefix_tokens["input_ids"])
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------------------
# Main
# ---------------------------

def main():
    """Main training function."""
    # --- Config ---
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    DATA_FILE = "transduction/train_dataset.json"
    MAX_LEN = 6000

    # Speed toggles (edit here or wire to argparse if you prefer)
    USE_COMPILE = True                   # torch.compile reduce-overhead
    FORCE_FLASH = True                   # try FlashAttention2 if installed
    MAX_TOKENS_PER_DEVICE = 0            # set e.g. 131072 to auto-tune microbatching
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 6
    USE_EARLY_STOP = False               # set True AND pass eval_dataset to enable

    # Mixed-precision prefs
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
    print(f"[info] bf16={'on' if use_bf16 else 'off'}")

    # Enable TF32 for extra throughput on NVIDIA (doesn't change numerics much)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Auth ---
    load_dotenv()
    if os.getenv("HF_TOKEN"):
        login(os.getenv("HF_TOKEN"))

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )
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

    # Keep memory in check
    model.gradient_checkpointing_enable()

    # --- LoRA (unchanged logic; still attention-side) ---
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # if you want full attention: ["q_proj","k_proj","v_proj","o_proj"]
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)  # type: ignore
    model.print_trainable_parameters()  # type: ignore

    # Compile after PEFT wraps & checkpointing are set
    model = maybe_compile(model, enable=USE_COMPILE, mode="reduce-overhead", fullgraph=False)

    # --- Dataset ---
    print(f"Loading dataset from {DATA_FILE}...")
    raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"Loaded {len(raw_ds)} training samples")

    def preprocess_fn(example):
        return preprocess_transduction_data(example, tokenizer, MAX_LEN)

    tokenised_ds = raw_ds.map(
        preprocess_fn,
        remove_columns=raw_ds.column_names,
        num_proc=max(1, os.cpu_count() // 2),
    )

    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # --- Training args ---
    # Start with your original microbatching, then optionally auto-tune by tokens.
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8

    if MAX_TOKENS_PER_DEVICE > 0:
        sizer = BatchSizer(
            max_tokens_per_device=MAX_TOKENS_PER_DEVICE,
            seq_len=MAX_LEN,
            starting_microbatch=per_device_train_batch_size,
        )
        per_device_train_batch_size, gradient_accumulation_steps = sizer.suggest(
            gradient_accumulation_steps=gradient_accumulation_steps
        )

    args = SFTConfig(
        output_dir="qwen3_30b_a3b_arc_transduction_sft",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
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
        optim="paged_adamw_8bit",  # 8-bit optimizer
        push_to_hub=True,
        hub_model_id="axel-darmouni/qwen2.5-0.5b-arc-transduction-sft",  # keep your ID (rename if you wish)
        **dataloader_kwargs(
            workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR,
        ),
    )

    # --- Trainer ---
    callbacks = []
    if USE_EARLY_STOP:
        callbacks.append(PlateauEarlyStop(monitor="eval_loss", min_delta=1e-3, patience=3))

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

    # --- Push to Hub (adapter weights) ---
    if os.getenv("HF_TOKEN"):
        print("Pushing adapter to Hugging Face Hub...")
        try:
            # Push PEFT adapter via trainer (safer for large base weights)
            trainer.push_to_hub()  # pushes to hub_model_id with adapter/config; not full base model
            print("Adapter pushed successfully!")
        except Exception as e:
            print(f"[warn] Push to hub failed: {e}")
    else:
        print("No HF_TOKEN found, skipping hub push")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
