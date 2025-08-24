"""Supervised Fine-Tuning script for ARC Transduction task."""

import os
import platform
from typing import Any, Dict, List
from dataclasses import dataclass

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    SFTConfig,
    SFTTrainer,
)


@dataclass
class DataCollatorForCausalLMWithPadding:
    """Data collator for causal language modeling with padding."""

    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8  # keep your 8-byte alignment
    label_pad_token_id: int = -100  # ignored by the loss

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Call the data collator."""
        # 1) pull out labels before calling tokenizer.pad()
        labels = [feat.pop("labels") for feat in features]

        # 2) pad input_ids & attention_mask
        batch = self.tokenizer.pad(  # type: ignore
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 3) pad labels to the same sequence length – fill with -100
        max_len = batch["input_ids"].size(1)
        batch["labels"] = torch.stack([
            torch.tensor(label + [self.label_pad_token_id] * (max_len - len(label)))
            for label in labels
        ])

        return batch  # type: ignore


def preprocess_transduction_data(example: Dict[str, Any], tokenizer: AutoTokenizer, max_len: int) -> Dict[str, Any]:
    """
    Preprocess transduction dataset for training.
    
    Args:
        example: Dictionary with 'input' (prompt) and 'output' (target) keys
        tokenizer: Tokenizer to use
        max_len: Maximum sequence length
        
    Returns:
        Processed example with input_ids, attention_mask, and labels
    """
    # Build messages for chat template
    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Apply chat template for full conversation
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Create prefix (everything up to assistant's response)
    prefix_messages = messages[:-1]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,  # ensures <|assistant|> is added
    )

    # Tokenize full and prefix (no padding during preprocessing)
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,  # Don't pad during preprocessing
    )
    prefix_tokens = tokenizer(
        prefix_text, 
        truncation=True, 
        max_length=max_len, 
        padding=False
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Create labels as a proper list copy
    labels = list(input_ids)  # Ensure it's a flat list
    prefix_len = len(prefix_tokens["input_ids"])

    # Mask all tokens up to the assistant's reply
    for i in range(min(prefix_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }


def main():
    """Main training function."""
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    DATA_FILE = "transduction/train_dataset.json"
    MAX_LEN = 2048
    
    # Load environment and login
    load_dotenv()
    login(os.getenv("HF_TOKEN"))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,  # needed for Qwen chat template
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with appropriate attention implementation
    attn_impl = "flash_attention_2" if platform.system() == "Linux" else "eager"

    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)  # type: ignore
    model.print_trainable_parameters()  # type: ignore

    # Load and preprocess dataset
    print(f"Loading dataset from {DATA_FILE}...")
    raw_ds = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"Loaded {len(raw_ds)} training samples")

    # Preprocess with partial function to pass additional arguments
    def preprocess_fn(example):
        return preprocess_transduction_data(example, tokenizer, MAX_LEN)

    tokenised_ds = raw_ds.map(
        preprocess_fn, 
        remove_columns=raw_ds.column_names, 
        num_proc=4
    )

    # Initialize data collator
    collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

    # Training arguments
    args = SFTConfig(
        output_dir="qwen2.5_0.5b_arc_transduction_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch size of 8
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=False,  # we're already in BF16
        bf16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        logging_dir="transduction/tb_logs",
        report_to="tensorboard",
        remove_unused_columns=False,
        # Note: Adjust DeepSpeed config path if needed
        deepspeed="transduction/training/ds_config_zero2.json",
        ddp_find_unused_parameters=False,
        push_to_hub=True,
        hub_model_id="axel-darmouni/qwen2.5-0.5b-arc-transduction-sft",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model, 
        args=args, 
        train_dataset=tokenised_ds, 
        data_collator=collator
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("qwen2.5_0.5b_arc_transduction_sft/final")

    # Push to hub if token is available
    if os.getenv("HF_TOKEN"):
        print("Pushing to Hugging Face Hub...")
        model.push_to_hub(
            "axel-darmouni/qwen2.5-0.5b-arc-transduction-sft",
            tokenizer=tokenizer,
            token=os.getenv("HF_TOKEN"),
        )
        print("Model pushed to hub successfully!")
    else:
        print("No HF_TOKEN found, skipping hub push")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
