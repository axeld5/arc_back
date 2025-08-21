"""RL script for training the transduction model."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    GSPOConfig,
    GSPOTrainer,
)

from transduction.training.reward_fn import reward_function


def transduction_reward_wrapper(
    completions: List[str], 
    prompts: List[str], 
    **kwargs: Any
) -> List[float]:
    """
    Wrapper function to adapt our transduction reward function for TRL.
    
    TRL expects a reward function that takes completions and prompts,
    but our reward function needs expected outputs. We'll extract the expected
    outputs from the dataset context or kwargs.
    
    Args:
        completions: List of model completions
        prompts: List of prompts (not used directly)
        **kwargs: Additional context, should contain 'expected_outputs'
        
    Returns:
        List of reward scores
    """
    # Extract expected outputs from kwargs
    expected_outputs = kwargs.get('expected_outputs', [])
    
    if not expected_outputs:
        # Fallback: return neutral rewards if no expected outputs provided
        return [0.0] * len(completions)
    
    return reward_function(completions, expected_outputs)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    login(os.getenv("HF_TOKEN"))
    
    # Model and data paths
    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    LORA_PATH = "qwen2.5_0.5b_arc_transduction_sft/final"  # SFT LoRA adapter
    DATA_PATH = "transduction/train_dataset.json"  # transduction dataset
    
    # ---------------------------------------------------------------------
    # 1. Tokenizer (ChatML template)
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---------------------------------------------------------------------
    # 2. Loading model (LoRA)
    # ---------------------------------------------------------------------
    attn_impl = "flash_attention_2" if platform.system() == "Linux" else "eager"
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        LORA_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to("cuda")
    
    # ---------------------------------------------------------------------
    # 3. Dataset ⇒ {"prompt", "expected_output"}
    # ---------------------------------------------------------------------
    raw_ds = load_dataset("json", data_files=DATA_PATH, split="train")
    
    def to_rl_format(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the transduction dataset to the RL format.
        
        The transduction dataset has 'input' (prompt) and 'output' (target) keys.
        We need to convert this to the format expected by TRL.
        """
        # Build messages for chat template
        messages = [
            {"role": "user", "content": example["input"]},
        ]
        
        # Create the prompt (without assistant response)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Parse the expected output string into a grid
        # The output is in semicolon-separated format like "12;34"
        expected_grid = []
        if example["output"]:
            try:
                rows = example["output"].split(';')
                for row in rows:
                    if row.strip():  # Skip empty rows
                        grid_row = [int(char) for char in row.strip()]
                        expected_grid.append(grid_row)
            except (ValueError, IndexError):
                # If parsing fails, create empty grid
                expected_grid = []
        
        return {
            "prompt": prompt,
            "expected_output": expected_grid,
            "original_input": example["input"],
            "original_output": example["output"]
        }
    
    ds = raw_ds.map(to_rl_format, remove_columns=raw_ds.column_names, num_proc=4)
    
    # ---------------------------------------------------------------------
    # 4. Custom reward function that uses dataset context
    # ---------------------------------------------------------------------
    def contextual_reward_function(
        completions: List[str], 
        prompts: List[str],
        **kwargs: Any
    ) -> List[float]:
        """
        Reward function that extracts expected outputs from the dataset batch.
        """
        # Get the current batch from the trainer context
        # This is a bit hacky but necessary for TRL integration
        batch = kwargs.get('batch', {})
        expected_outputs = batch.get('expected_output', [])
        
        if not expected_outputs:
            # Fallback: return neutral rewards
            print("Warning: No expected outputs found in batch context")
            return [0.0] * len(completions)
        
        # Convert to list format if needed
        if isinstance(expected_outputs, torch.Tensor):
            expected_outputs = expected_outputs.tolist()
        
        return reward_function(completions, expected_outputs)
    
    # ---------------------------------------------------------------------
    # 5. GSPO config with transduction-specific parameters
    # ---------------------------------------------------------------------
    gspo_cfg = GSPOConfig(
        output_dir="qwen2.5_0.5b_arc_transduction_rl",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller for 0.5B model
        num_train_epochs=1,
        learning_rate=1e-5,  # Lower learning rate for RL
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        logging_dir="transduction/rl_tb_logs",
        report_to="tensorboard",
        # GSPO-specific parameters
        num_generations=4,  # G in the paper
        max_prompt_length=2048,  # Match SFT max length
        max_completion_length=1024,  # Enough for transduction outputs
        remove_unused_columns=False,  # Keep expected_output
        push_to_hub=True,
        hub_model_id="axel-darmouni/qwen2.5-0.5b-arc-transduction-rl",
        # Uncomment if using DeepSpeed
        deepspeed="transduction/training/ds_config_zero2.json",
        ddp_find_unused_parameters=False,
    )
    
    # ---------------------------------------------------------------------
    # 6. Trainer
    # ---------------------------------------------------------------------
    trainer = GSPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[contextual_reward_function],
        args=gspo_cfg,
        train_dataset=ds,
    )
    
    print("Starting RL training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("qwen2.5_0.5b_arc_transduction_rl/final")
    
    # Optional: Save to hub
    if os.getenv("HF_TOKEN"):
        print("Pushing to Hugging Face Hub...")
        model.push_to_hub(
            "axel-darmouni/qwen2.5-0.5b-arc-transduction-rl",
            tokenizer=tokenizer,
            token=os.getenv("HF_TOKEN"),
        )
        print("Model pushed to hub successfully!")
    else:
        print("No HF_TOKEN found, skipping hub push")
    
    print("RL training completed successfully!")
