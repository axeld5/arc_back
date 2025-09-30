import json
import pandas as pd
from typing import *
from loader import load_training_problem, list_training_problems, load_evaluation_problem, list_evaluation_problems

import unsloth
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from datasets import Dataset

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

PROMPT_INDUCTION = (
    "Solve the following problem\n\n"
    "Given input/output pairs:\n{io_pairs}\n"
    "Write a python program that solves the problem. Name your final function 'p'.\n"
    "OUTPUT:"
)

def format_comparison(output_array, predicted_output):
    expected_str = '\n'.join(' '.join(map(str, row)) for row in output_array)
    got_str = '\n'.join(' '.join(map(str, row)) for row in predicted_output)
    expected_lines = expected_str.split('\n')
    got_lines = got_str.split('\n')
    max_lines = max(len(expected_lines), len(got_lines))
    comparison = []
    for i in range(max_lines):
        expected_line = expected_lines[i] if i < len(expected_lines) else ""
        got_line = got_lines[i] if i < len(got_lines) else ""
        comparison.append(f"{got_line} -> {expected_line}")
    return comparison

def evaluate_prediction(input_array, output_array, response, debug=False):
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    try:
        start_marker = "```python"
        end_marker = "```"
        start_idx = response.find(start_marker)
        if start_idx == -1:
            if debug:
                print(f"No Python code block found in response")
            return False
        start_idx += len(start_marker)
        end_idx = response.find(end_marker, start_idx)
        if end_idx == -1:
            if debug:
                print(f"No closing code block marker found")
            return False
        code = response[start_idx:end_idx].strip()
        
        # Set up timeout for code execution
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(90)  # 1 minute 30 seconds timeout
        
        try:
            local_namespace = {}
            exec(code, local_namespace)
            if 'p' not in local_namespace:
                if debug:
                    print(f"Function 'p' not found in generated code")
                signal.alarm(0)  # Cancel the alarm
                return False
            predicted_output = local_namespace['p'](input_array)
            signal.alarm(0)  # Cancel the alarm
            
            if predicted_output == output_array:
                if debug:
                    print(f"✓ Correct prediction for input/output pair")
                return True
            else:
                if debug:
                    print(f"✗ Incorrect prediction for input/output pair")
                    comparison = format_comparison(output_array, predicted_output)
                    print(f"Comparison (Got -> Expected):\n" + '\n'.join(comparison))
                return False
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            if debug:
                print(f"Code execution timed out after 90 seconds")
            return False
            
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        if debug:
            print(f"Error executing generated code: {e}")
            print(f"Generated code was: {code if 'code' in locals() else 'N/A'}")
        return False

def grid_to_row_strings(grid: List[List[int]]) -> List[str]:
    return [' '.join(map(str, row)) for row in grid]

def _format_induction_prompt(problem) -> str:
    input_output_pairs = ""
    for i, elem in enumerate(problem['train']):
        pb_input ="\n".join(grid_to_row_strings(elem['input']))
        pb_output = "\n".join(grid_to_row_strings(elem['output']))
        input_output_pairs += f"Input {i+1}:\n{pb_input}\nOutput {i+1}:\n{pb_output}\n\n"
    return PROMPT_INDUCTION.format(io_pairs=input_output_pairs)

def _format_code_solution(problem_id):
    reasoning_path = f"reasoning_files/{problem_id}.txt"
    with open(reasoning_path, 'r', encoding='utf-8') as f:
        reasoning = f.read()
    solver_path = f"remapped_solvers/{problem_id}.py"
    with open(solver_path, 'r', encoding='utf-8') as f:
        solver_code = f.read()
    solution = f"""<think>
    {reasoning}
    </think>
    Here's the code that solves the problem:
    ```python
    {solver_code}
    ```"""
    return solution

def get_data(max_samples: Optional[int] = None):
    train_problems = {"conversations":[], "arrays":[]}
    data = list_training_problems()
    if max_samples is None:
        max_samples = len(data)
    for problem_id in data[:max_samples]:
        print(f"Processing problem {problem_id}")
        problem = load_training_problem(problem_id)
        user_content = {"role":"user", "content":""}
        user_content["content"] = _format_induction_prompt(problem)
        assistant_content = {"role":"assistant", "content":""}
        assistant_content["content"] = _format_code_solution(problem_id)
        train_problems["conversations"].append([user_content, assistant_content])
        train_problems["arrays"].append(problem["train"])

    test_problems = {"conversations":[], "arrays":[]}
    eval_data = list_evaluation_problems()
    for problem_id in eval_data:
        problem = load_evaluation_problem(problem_id)
        user_content = {"role":"user", "content":""}
        user_content["content"] = _format_induction_prompt(problem)
        test_problems["conversations"].append([user_content])
        test_problems["arrays"].append(problem["train"])

    with open('data.json', 'w') as f:
        json.dump(train_problems, f)
    with open('test_problems.json', 'w') as f:
        json.dump(test_problems, f)

def config_data_for_sft(dataset_path: str, tokenizer):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    formatted_data = tokenizer.apply_chat_template(
        data["conversations"],
        tokenize = False,
    )
    formatted_data = pd.Series(formatted_data)
    formatted_data.name = "text"
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset

def run_sft(
    dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_sft",
    base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct",
    learning_rate: float = 5e-4,
    num_train_epochs: int = 10,
):      
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,                 # use the arg instead of hardcoding
        max_seq_length = 20000,
        dtype = compute_dtype,
        load_in_4bit = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 256,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
    )
    dataset = config_data_for_sft(dataset_path, tokenizer)
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_8bit",
        ddp_find_unused_parameters=False,
        max_grad_norm=None,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    print("[sft] Starting training...")
    trainer.train()
    print("[sft] Saving final adapter...")
    model_save_path = os.path.join(output_dir, "final")
    merged_save_path = os.path.join(output_dir, "merged")
    model.save_pretrained(model_save_path)
    try:
        tokenizer.save_pretrained(model_save_path)
        model.save_pretrained_merged(merged_save_path, tokenizer, save_method = "merged_16bit",)
    except Exception:
        pass    
    return model_save_path, merged_save_path

"""
from unsloth import FastLanguageModel
base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-Coder-3B-Instruct",
        max_seq_length = 20000,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "qwen3_4b_singled_out_sft/final")
FastLanguageModel.for_inference(model)
with open("data.json") as f:
    raw = json.load(f)

total_valid = 0
for k in range(len(raw["conversations"])):
    print(f"Processing problem {k}")
    sample_data = raw["conversations"][k][0]["content"]
    messages = [
        {"role": "user", "content": sample_data},
    ]
    #prompts = [sample_data]*10
    arrays = raw["arrays"][k]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")
    #decoded = llm.generate(prompts, sampling_params)
    for p in range(5):
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Generation timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(90)  # 90 second timeout
        
        try:
            outputs = model.generate(input_ids = inputs, max_new_tokens = 5000,  
            temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, use_cache = True)
            generated_tokens = outputs[:, inputs.shape[-1]:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            code_resolution = decoded[0]
            signal.alarm(0)  # Cancel the alarm
            
            cnt = 0
            for inp_out in arrays:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                if evaluate_prediction(input_array, output_array, code_resolution, debug=True):
                    cnt += 1
            if cnt == len(arrays):
                print(f"✓ problem {k}")
                total_valid += 1
                break
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            print(f"⏰ Generation timed out for problem {k}")
            continue
        else:
            print(f"✗ problem {k}")
print(f"Total valid: {total_valid}/{len(raw['conversations'])}")
"""

def evaluate_code_validity(
    completions: List[str],
    arrays: List[str],
    **kwargs: Any
) -> List[float]:
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
        for inp_out in array_list:
            input_array = inp_out["input"]
            output_array = inp_out["output"]
            if not evaluate_prediction(input_array, output_array, value):
                rewards.append(-0.5)
                break
        else:
            rewards.append(1.0)
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
    output_dir: str = "qwen3_4b_singled_out_rl",
    learning_rate: float = 5e-4,
    num_train_epochs: int = 1,
    grad_accum: int = 4,
    num_generations: int = 8,
):
    max_seq_length = 30000
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = sft_merged_save_path,
        max_seq_length = max_seq_length,
        dtype = compute_dtype,
        load_in_4bit = True,
        fast_inference = True, # Enable vLLM fast inference
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth"
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
    training_args = GRPOConfig(
        #use_vllm=True,
        #importance_sampling_level="sequence",
        #loss_type="grpo",
        vllm_sampling_params=vllm_sampling_params,
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Reduced from 8 to help with memory
        gradient_accumulation_steps=grad_accum,
        beta=0.04,
        epsilon=3e-4,
        max_steps=500,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
        num_generations=num_generations,
        max_prompt_length=12000,  # Reduced from 20000
        max_completion_length=6000,  # Reduced from 8192 (total ~18k < 20k model limit)
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
    trainer.save_model(os.path.join(output_dir, "final"))
    try:
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    except Exception:
        pass
    return os.path.join(output_dir, "final")

def inference_loop(model_path: str, base_model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct"):
    from peft import PeftModel
    
    # Load base model and tokenizer
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = 20000,
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
        load_in_4bit = True,
    )
    
    # Load fine-tuned model
    model = PeftModel.from_pretrained(base_model, model_path)
    FastLanguageModel.for_inference(model)
    
    # Load data
    with open("data.json") as f:
        raw = json.load(f)
    
    total_valid = 0
    for k in range(len(raw["conversations"])):
        print(f"Processing problem {k}")
        sample_data = raw["conversations"][k][0]["content"]
        messages = [
            {"role": "user", "content": sample_data},
        ]
        arrays = raw["arrays"][k]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")
        for p in range(10):
            outputs = model.generate(input_ids = inputs, max_new_tokens = 5000,  
            temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, use_cache = True)
            generated_tokens = outputs[:, inputs.shape[-1]:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            code_resolution = decoded[0]
            cnt = 0
            for inp_out in arrays:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                if evaluate_prediction(input_array, output_array, code_resolution, debug=True):
                    cnt += 1
            if cnt == len(arrays):
                print(f"✓ problem {k}")
                total_valid += 1
                break
            else:
                print(f"✗ problem {k}")
    print(f"Total valid: {total_valid}/{len(raw['conversations'])}")


if __name__ == "__main__":
    #get_data()
    #sft_model_save_path, sft_merged_save_path = run_sft("data.json")
    sft_merged_save_path = "qwen3_4b_singled_out_sft/merged"
    model_save_path = run_rl(sft_merged_save_path)
    inference_loop(model_save_path)