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

PatchFastRL("GRPO", FastLanguageModel)

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

def evaluate_code_validity(
    completions: List[str],
    arrays: List[str],
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
            all_correct = True
            for inp_out in array_list:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                
                try:
                    local_namespace = {}
                    exec(code, local_namespace)
                    if 'p' not in local_namespace:
                        all_correct = False
                        break
                    predicted_output = local_namespace['p'](input_array)
                    print(predicted_output)
                    print(output_array)
                    if predicted_output != output_array:
                        all_correct = False
                        break
                except Exception:
                    all_correct = False
                    break
            
            signal.alarm(0)  # Cancel the alarm
            
            if all_correct:
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
    output_dir: str = "gptoss_induction_rl",
    learning_rate: float = 5e-5,
    num_train_epochs: int = 1,
    grad_accum: int = 4,
    num_generations: int = 4,
):
    max_seq_length = 30000
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = sft_merged_save_path,
        max_seq_length = max_seq_length,
        dtype = compute_dtype,
        load_in_4bit = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = False
    )
    with open("data.json") as f:
        raw = json.load(f)
    converted = convert_conversations(raw)
    dataset = Dataset.from_list(converted)
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    print("starting training")
    training_args = GRPOConfig(
        #use_vllm=True,
        #vllm_mode="server",
        #vllm_server_host="127.0.0.1",
        #vllm_server_port=8000,
        importance_sampling_level="sequence",
        loss_type="grpo",
        #vllm_sampling_params=vllm_sampling_params,
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Reduced from 8 to help with memory
        gradient_accumulation_steps=grad_accum,
        beta=0.04,
        epsilon=3e-4,
        max_steps=100,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=50,
        optim="paged_adamw_8bit",
        report_to="none",
        num_generations=2,
        max_prompt_length=15000,  # Reduced from 20000
        max_completion_length=8192,  # Reduced from 8192 (total ~18k < 20k model limit)
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [evaluate_code_validity],
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
        if os.getenv("HF_TOKEN"):
            model.push_to_hub_merged("axel-darmouni/gptoss-induction-sft", tokenizer, save_method = "merged_16bit", token = os.getenv("HF_TOKEN"))
    except Exception:
        pass
    return os.path.join(output_dir, "final")


if __name__ == "__main__":
    sft_merged_save_path = "gptoss_induction_sft/merged"
    model_save_path = run_rl(sft_merged_save_path)
    print(f"RL model saved to: {model_save_path}")

