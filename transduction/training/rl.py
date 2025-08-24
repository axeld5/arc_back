"""RL script for training the transduction model."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import (
    GRPOConfig,
    GRPOTrainer,
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
    BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    LORA_PATH = "qwen2.5_0.5b_arc_transduction_sft/final"  # SFT LoRA adapter
    DATA_PATH = "transduction/train_dataset.json"  # transduction dataset
    
    # ---------------------------------------------------------------------
    # 1. Tokenizer (ChatML template)
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---------------------------------------------------------------------
    # 2. Loading model (Base model + LoRA adapter)
    # ---------------------------------------------------------------------
    attn_impl = "flash_attention_2" if platform.system() == "Linux" else "eager"
    
    # Configure quantization: 8-bit model loading
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )
    
    # First load the base model with 8-bit quantization
    base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    
    # Then load the LoRA adapter (will be loaded in 4-bit by default with quantized base model)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Ensure only LoRA parameters require gradients for RL training
    model.train()
    trainable_params = 0
    total_params = 0
    
    # First, disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(False)
        total_params += param.numel()
    
    # Then enable gradients only for LoRA parameters (floating point tensors)
    lora_keywords = ['lora_', 'adapter', 'peft', 'modules_to_save']
    for name, param in model.named_parameters():
        # Check if this is a LoRA parameter
        is_lora_param = any(keyword in name.lower() for keyword in lora_keywords)
        
        if is_lora_param:
            # Only enable gradients for floating point parameters
            if param.dtype.is_floating_point:
                param.requires_grad_(True)
                trainable_params += param.numel()
                print(f"Enabled gradients for LoRA parameter: {name} (dtype: {param.dtype})")
            else:
                print(f"Skipping non-floating point LoRA parameter: {name} (dtype: {param.dtype})")
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Verify we have trainable parameters
    if trainable_params == 0:
        print("Warning: No trainable parameters found! This might indicate an issue with LoRA adapter loading.")
        print("Trying alternative approach using PEFT methods...")
        
        # Alternative: Use PEFT's get_trainable_parameters method if available
        try:
            if hasattr(model, 'get_trainable_parameters'):
                trainable_param_names = model.get_trainable_parameters()
                for name, param in model.named_parameters():
                    if name in trainable_param_names and param.dtype.is_floating_point:
                        param.requires_grad_(True)
                        trainable_params += param.numel()
                        print(f"Enabled gradients via PEFT method: {name}")
            else:
                # Final fallback: enable gradients for any floating point parameters with reasonable size
                for name, param in model.named_parameters():
                    if (param.dtype.is_floating_point and 
                        param.numel() < 10000000 and  # Less than 10M parameters
                        ('weight' in name.lower() or 'bias' in name.lower())):
                        param.requires_grad_(True)
                        trainable_params += param.numel()
                        print(f"Enabled gradients (fallback): {name}")
        except Exception as e:
            print(f"Error in fallback gradient setup: {e}")
        
        print(f"Updated trainable parameters: {trainable_params:,}")
        
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters found! Cannot proceed with RL training.")
    
    # Model is already on correct device due to device_map="auto"
    
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
    # 4. Create a closure-based reward function with dataset access
    # ---------------------------------------------------------------------
    
    # Create a mapping from prompts to expected outputs for fast lookup
    prompt_to_expected = {}
    for example in ds:
        prompt_to_expected[example["prompt"]] = example["expected_output"]
    
    print(f"Created prompt-to-expected mapping with {len(prompt_to_expected)} entries")
    
    def contextual_reward_function(
        completions: List[str], 
        prompts: List[str],
        **kwargs: Any
    ) -> List[float]:
        """
        Reward function that looks up expected outputs based on prompts.
        """
        expected_outputs = []
        
        for prompt in prompts:
            # Look up the expected output for this prompt
            expected_output = prompt_to_expected.get(prompt, [])
            expected_outputs.append(expected_output)
        
        if not any(expected_outputs):
            # Fallback: return neutral rewards if no expected outputs found
            print(f"Warning: No expected outputs found for {len(prompts)} prompts")
            return [0.0] * len(completions)
        
        # Debug: Print first few examples on first call
        if not hasattr(contextual_reward_function, '_debug_printed'):
            print(f"Debug: First reward function call with {len(completions)} completions")
            if completions and expected_outputs:
                print(f"Debug: First completion: {completions[0][:50]}...")
                print(f"Debug: First expected output shape: {len(expected_outputs[0]) if expected_outputs[0] else 0}")
            contextual_reward_function._debug_printed = True
        
        # Get rewards and ensure they are Python floats
        rewards = reward_function(completions, expected_outputs)
        return [float(r) for r in rewards]
    
    # ---------------------------------------------------------------------
    # 5. GSPO config with transduction-specific parameters
    # ---------------------------------------------------------------------
    gspo_cfg = GRPOConfig(
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir="qwen2.5_0.5b_arc_transduction_rl",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller for 0.5B model
        beta=0.04,  # not explicitly specified in the paper, but they likely used the same value as in the GRPO paper
        epsilon=3e-4,  # https://x.com/ChujieZheng/status/1948933507696525392
        num_train_epochs=1,
        learning_rate=1e-5,  # Lower learning rate for RL
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        logging_dir="transduction/rl_tb_logs",
        report_to="tensorboard",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.30,
        # GSPO-specific parameters
        num_generations=4,  # G in the paper
        max_prompt_length=6000,  # Match SFT max length
        max_completion_length=1024,  # Enough for transduction outputs
        remove_unused_columns=False,  # Keep expected_output
        push_to_hub=True,
        hub_model_id="axel-darmouni/qwen2.5-0.5b-arc-transduction-rl",
        # Uncomment if using DeepSpeed
        #deepspeed="transduction/training/ds_config_zero2.json",
        #ddp_find_unused_parameters=False,
    )
    
    # ---------------------------------------------------------------------
    # 6. Trainer
    # ---------------------------------------------------------------------
    trainer = GRPOTrainer(
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
