import json
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import unsloth
import torch
from typing import List, Any
from dotenv import load_dotenv
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL
from datasets import Dataset
from accelerate import Accelerator
accel = Accelerator(device_placement=False)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)                 # <- critical
device_map = {"": local_rank}                     # <- one GPU per rank

class PatchedGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure DDP-wrapped models expose .config
        try:
            _ = model.config  # will raise on DDP
        except AttributeError:
            base = self.accelerator.unwrap_model(model)
            # Attach the underlying config to the wrapper so downstream code works
            object.__setattr__(model, "config", getattr(base, "config", None))
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

def evaluate_code_validity(
    completions: List[str],
    arrays: List[str],
    is_partial_rl: bool = False,
    **kwargs: Any
) -> List[float]:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    rewards: List[float] = []
    for completion, array_list in zip(completions, arrays, strict=False):
        value = completion[0]["content"]
        start_marker = "```python"
        end_marker = "```"
        start_idx = value.find(start_marker)
        if start_idx == -1:
            rewards.append(-1.0)
            continue
        start_idx += len(start_marker)
        end_idx = value.find(end_marker, start_idx)
        if end_idx == -1:
            rewards.append(-1.0)
            continue
        
        code = value[start_idx:end_idx].strip()
        
        # Set up timeout for code execution
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        try:
            n_examples = len(array_list)
            n_examples_solved = 0
            
            for inp_out in array_list:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                
                try:
                    local_namespace = {}
                    exec(code, local_namespace)
                    if 'transform' not in local_namespace:
                        break
                    predicted_output = local_namespace['transform'](input_array)
                    if predicted_output == output_array:
                        n_examples_solved += 1
                except Exception:
                    pass  # Continue checking other examples in partial mode
            
            signal.alarm(0)  # Cancel the alarm
            
            if is_partial_rl:
                # Partial reward: proportional to examples solved
                if n_examples > 0:
                    rewards.append(n_examples_solved / n_examples)
                else:
                    rewards.append(0.0)
            else:
                # Binary reward: all or nothing
                if n_examples_solved == n_examples:
                    rewards.append(1.0)
                else:
                    rewards.append(-0.5)
                
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            rewards.append(-1.0)  # Return -1 for timeout
    
    return rewards

def convert_conversations(raw_json):
    result = []
    for convo, array_list in zip(raw_json["conversations"], raw_json["arrays"]):
        user_msg = convo[0]["content"]
        result.append({
            "prompt": [
                {"role": "user", "content": user_msg}
            ],
            "arrays": array_list
        })
    return result

def run_rl(
    sft_merged_save_path: str,
    output_dir: str = "qwen2.5_7b_singled_out_rl",
    learning_rate: float = 5e-5,
    num_steps: int = 100,
    grad_accum: int = 2,
    num_generations: int = 2,
    data_dir: str = "data.json",
    is_partial: bool = False,
):
    max_seq_length = 15000
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = sft_merged_save_path,
        max_seq_length = max_seq_length,
        dtype = compute_dtype,
        load_in_4bit = False,        # <- be explicit
        fast_inference = False,      # <- ensure no embedded vLLM
        device_map = device_map,     # <- pin to the per-rank GPU
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 1,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = False
    )
    with open(data_dir) as f:
        raw = json.load(f)
    converted = convert_conversations(raw)
    dataset = Dataset.from_list(converted)
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    training_args = GRPOConfig(
        #use_vllm=True,
        vllm_mode="server",
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,
        #importance_sampling_level="sequence",
        #loss_type="grpo",
        vllm_sampling_params=vllm_sampling_params,
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Reduced from 8 to help with memory
        gradient_accumulation_steps=grad_accum,
        beta=0.04,
        epsilon=3e-4,
        max_steps=num_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
        num_generations=num_generations,
        max_prompt_length=10000,  # Reduced from 20000
        max_completion_length=2048,  # Reduced from 8192 (total ~18k < 20k model limit)
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # Create reward function with is_partial_rl parameter
    def reward_func_with_partial(completions, arrays, **kwargs):
        return evaluate_code_validity(completions, arrays, is_partial_rl=is_partial, **kwargs)
    
    trainer = PatchedGRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [reward_func_with_partial],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    model.save_pretrained(model_save_path)
    try:
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)
        if os.getenv("HF_TOKEN") and not is_partial:
            model.push_to_hub_merged("axel-darmouni/qwen2.5-7b-soar-induction-rl", tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))
    except Exception:
        pass
    return model_save_path


if __name__ == "__main__":    
    sft_merged_save_path = "qwen2.5_7b_singled_out_sft/merged"
    
    print("=" * 80)
    print("Stage 1: Training with PARTIAL reward function")
    print("=" * 80)
    
    # First stage: Train with partial reward (proportional to examples solved)
    stage1_path = run_rl(
        sft_merged_save_path=sft_merged_save_path,
        output_dir="qwen2.5_7b_induction_rl_partial",
        learning_rate=5e-5,
        num_steps=100,
        grad_accum=4,
        num_generations=2,
        data_dir="full_set.json",
        is_partial=True,
    )
    
    print("\n" + "=" * 80)
    print("Stage 2: Training with FULL reward function (binary)")
    print("=" * 80)
    
    # Second stage: Train with full binary reward on the model from stage 1
    stage2_path = run_rl(
        sft_merged_save_path=stage1_path,
        output_dir="qwen2.5_7b_induction_rl_full",
        learning_rate=5e-5,
        num_steps=400,
        grad_accum=4,
        num_generations=2,
        data_dir="test_problems.json",
        is_partial=False,
    )
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Stage 1 (partial) model saved at: {stage1_path}")
    print(f"Stage 2 (full) model saved at: {stage2_path}")
    print("=" * 80)

