import json
from typing import *
from loader import load_training_problem, list_training_problems


import unsloth
import os
import platform
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

load_dotenv()
if os.getenv("HF_TOKEN"):
    try:
        login(os.getenv("HF_TOKEN"))
    except Exception:
        pass

PROMPT_INDUCTION = (
    "Solve the following problem\n\n"
    "Given input/output pairs:\n{input}\n"
    "Write a python program that solves the problem. Name your final function 'p'.\n"
    "OUTPUT:"
)

def grid_to_row_strings(grid: List[List[int]]) -> List[str]:
    return [' '.join(map(str, row)) for row in grid]

def _format_induction_prompt(problem) -> str:
    input_output_pairs = ""
    for i, elem in enumerate(problem['train']):
        pb_input ="\n".join(grid_to_row_strings(elem['input']))
        pb_output = "\n".join(grid_to_row_strings(elem['output']))
        input_output_pairs += f"Input {i+1}:\n{pb_input}\nOutput {i+1}:\n{pb_output}\n\n"
    return PROMPT_INDUCTION.format(input=input_output_pairs)

def _format_code_solution(problem_id):
    reasoning_path = f"reasoning_files/{problem_id}.txt"
    with open(reasoning_path, 'r') as f:
        reasoning = f.read()
    solver_path = f"remapped_solvers/{problem_id}.py"
    with open(solver_path, 'r') as f:
        solver_code = f.read()
    solution = f"""{reasoning}
    Here's the code that solves the problem:
    ```python
    {solver_code}
    ```"""
    return solution

train_problems = {"conversations":[]}
data = list_training_problems()
problem_id = data[0]
problem = load_training_problem(problem_id)
user_content = {"role":"user", "content":""}
user_content["content"] = _format_induction_prompt(problem)
assistant_content = {"role":"assistant", "content":""}
assistant_content["content"] = _format_code_solution(problem_id)
train_problems["conversations"].append([user_content, assistant_content])
print(train_problems)
"""
with open('data.json', 'w') as f:
    json.dump(train_problems, f)
with open('test_problems.json', 'w') as f:
    json.dump(test_problems, f)

def pick_attn_impl() -> str:
    if platform.system() == "Linux":
        try:
            import importlib
            importlib.import_module("flash_attn")
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    return "sdpa"

def run_sft(
    dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_sft",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    learning_rate: float = 8e-5,
    num_train_epochs: int = 300,
    use_compile: bool = False,
):      
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    attn_impl = pick_attn_impl()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length = 8192,
        dtype = compute_dtype,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  
        lora_dropout = 0, 
        bias = "none",     
        use_gradient_checkpointing = "unsloth", 
    )
    with open("data.json") as f:
        raw = json.load(f)
    data = tokenizer.apply_chat_template(
        raw["conversations"],
        tokenize = False,
    )
    import pandas as pd
    data = pd.Series(data)
    data.name = "text"
    
    from datasets import Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    dataset = dataset.shuffle(seed = 3407)

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
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
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=8192,
    )
        
    print("[sft] Starting training...")
    trainer.train()
    print("[sft] Saving final adapter...")
    trainer.save_model(os.path.join(output_dir, "final"))
    try:
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    except Exception:
        pass    
    return os.path.join(output_dir, "final")

run_sft("data.json")

import re
from typing import List, Optional


def check_array(output_string: str) -> bool:
    if not output_string or not isinstance(output_string, str):
        return False
    response = output_string.strip()
    if not response:
        return False
    if '\n' in response:
        grid_match = re.search(r'[0-9\n\s]+', response)
        if not grid_match:
            return False
        grid_str = grid_match.group()
        try:
            rows = grid_str.split('\n')
            if not rows:
                return False
            grid = []
            expected_width = None
            for row in rows:
                if not row.strip():
                    return False
                parts = row.strip().split()
                if len(parts) > 1:
                    try:
                        grid_row = [int(p) for p in parts if p.strip()]
                    except ValueError:
                        return False
                else:
                    if not row.strip().isdigit():
                        return False
                    grid_row = [int(char) for char in row.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return False
                if expected_width is None:
                    expected_width = len(grid_row)
                elif len(grid_row) != expected_width:
                    return False
                grid.append(grid_row)
            return len(grid) > 0 and len(grid[0]) > 0
        except (ValueError, IndexError):
            return False
    return False


def check_value(output_string: str, expected_value: List[List[int]]) -> bool:
    if not isinstance(expected_value, list) or not expected_value:
        return False
    if not check_array(output_string):
        return False
    parsed_grid = parse_grid_from_string(output_string)
    if parsed_grid is None:
        return False
    return parsed_grid == expected_value

def same_shape(a: List[List[int]], b: List[List[int]]) -> bool:
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) for ra, rb in zip(a, b))


def parse_grid_from_string(output_string: str) -> Optional[List[List[int]]]:
    if not output_string or not isinstance(output_string, str):
        return None
    response = output_string.strip()
    if not response:
        return None
    if '\n' in response:
        grid_match = re.search(r'[0-9\n\s]+', response)
        if not grid_match:
            return None
        grid_str = grid_match.group()
        try:
            rows = grid_str.split('\n')
            grid = []
            for row in rows:
                if not row.strip():
                    continue
                parts = row.strip().split()
                if len(parts) > 1:
                    try:
                        grid_row = [int(p) for p in parts if p.strip()]
                    except ValueError:
                        return None
                else:
                    if not row.strip().isdigit():
                        return None
                    grid_row = [int(char) for char in row.strip()]
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return None
                grid.append(grid_row)
            return grid if grid else None
        except (ValueError, IndexError):
            return None
    return None

def reward_function(
    completions: List[str], 
    expected_output: List[str], 
    **kwargs: Any
) -> List[float]:
    rewards = []
    for completion, expected in zip(completions, expected_output, strict=False):
        if not check_array(completion):
            rewards.append(-1.0)
            continue
        if check_value(completion, parse_grid_from_string(expected)):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

def reward_function_diff(
    completions: List[str],
    expected_output: List[str],
    **kwargs: Any
) -> List[float]:
    rewards: List[float] = []
    for completion, expected in zip(completions, expected_output, strict=False):
        if not check_array(completion) or not check_array(expected):
            rewards.append(-1.0)
            continue

        comp_grid = parse_grid_from_string(completion)
        exp_grid = parse_grid_from_string(expected)
        if comp_grid is None or exp_grid is None or not same_shape(comp_grid, exp_grid):
            rewards.append(-1.0)
            continue

        rows = len(exp_grid)
        cols = len(exp_grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            rewards.append(-0.5)
            continue

        diffs = 0
        for r in range(rows):
            for c in range(cols):
                if comp_grid[r][c] != exp_grid[r][c]:
                    diffs += 1

        rewards.append(1 - diffs / (rows * cols))
    return rewards

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_4b_singled_out_sft/final", 
        max_seq_length = 8192,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )

model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is 
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
)

with open("data.json") as f:
        raw = json.load(f)
sample_data = raw["conversations"][0][0]["content"]
messages = [
            {"role": "user", "content": sample_data},
]
from unsloth.chat_templates import get_chat_template
FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_array(decoded[0]))
print(check_value(decoded[0], parse_grid_from_string(raw["conversations"][0][1]["content"])))
print(check_value(raw["conversations"][0][1]["content"], parse_grid_from_string(raw["conversations"][0][1]["content"])))
print(reward_function(decoded, [raw["conversations"][0][1]["content"]]))

def reward_function(
    completions: List[str], 
    expected_output: List[str], 
    **kwargs: Any
) -> List[float]:
    rewards = []
    for completion, expected in zip(completions, expected_output, strict=False):
        value = completion[0]["content"]
        if not check_array(value):
            rewards.append(-1.0)
            continue
        if check_value(value, parse_grid_from_string(expected)):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

def reward_function_diff(
    completions: List[str],
    expected_output: List[str],
    **kwargs: Any
) -> List[float]:
    rewards: List[float] = []
    for completion, expected in zip(completions, expected_output, strict=False):
        value = completion[0]["content"]
        if not check_array(value):
            rewards.append(-1.0)
            continue
        comp_grid = parse_grid_from_string(value)
        exp_grid = parse_grid_from_string(expected)
        if comp_grid is None or exp_grid is None or not same_shape(comp_grid, exp_grid):
            rewards.append(-0.5)
            continue
        rows = len(exp_grid)
        cols = len(exp_grid[0]) if rows else 0
        diffs = 0
        for r in range(rows):
            for c in range(cols):
                if comp_grid[r][c] != exp_grid[r][c]:
                    diffs += 1
        if diffs != 0:
            rewards.append(0.5 * (1 - diffs / (rows * cols)))
        else:
            rewards.append(1)
    return rewards

def convert_conversations(raw_json):
    result = []
    for convo in raw_json["conversations"]:
        # Expecting [ {"role":"user"}, {"role":"assistant"} ]
        user_msg = convo[0]["content"]
        assistant_msg = convo[1]["content"]
        result.append({
            "prompt": [
                {"role": "user", "content": user_msg}
            ],
            "expected_output": assistant_msg
        })
    return result

def run_rl(
    #base_model: str,
    #lora_path: str,
    #dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_rl",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    grad_accum: int = 4,
    num_generations: int = 4,
):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 8192 # Can increase for longer reasoning traces
    lora_rank = 128 # Larger rank = smarter, but slower
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_4b_singled_out_sft/final",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.2, # Reduce if out of memory
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is 
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    )
    with open("data.json") as f:
        raw = json.load(f)
    converted = convert_conversations(raw)
    dataset = Dataset.from_list(converted)  
    print(dataset)
    print("num examples:", len(dataset))  # should be > 0
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    
    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        use_vllm=True,
        importance_sampling_level="sequence",
        loss_type="grpo",
        output_dir=output_dir,
        per_device_train_batch_size=1,
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
        num_generations=4,
        max_prompt_length=4096,
        max_completion_length=2048,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_function_diff
        ],
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

run_rl()

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_4b_singled_out_rl/final", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = 8192,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        fast_inference = True,
    )

model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is 
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
)

with open("data.json") as f:
        raw = json.load(f)
sample_data = raw["conversations"][0][0]["content"]
messages = [
            {"role": "user", "content": sample_data},
]
from unsloth.chat_templates import get_chat_template
FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_array(decoded[0]))
print(check_value(decoded[0], parse_grid_from_string(raw["conversations"][0][1]["content"])))

sample_data = raw["conversations"][1][0]["content"]
messages = [
            {"role": "user", "content": sample_data},
]
from unsloth.chat_templates import get_chat_template
FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_value(decoded[0], parse_grid_from_string(raw["conversations"][1][1]["content"])))


sample_data = test_problems["conversations"][0][0]["content"]
messages = [
            {"role": "user", "content": sample_data},
]
from unsloth.chat_templates import get_chat_template
FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_array(decoded[0]))
print(check_value(decoded[0], parse_grid_from_string(test_problems["conversations"][0][1]["content"])))

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_4b_singled_out_sft/final", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = 8192,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        fast_inference = True,
    )

model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is 
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
)

sample_data = test_problems["conversations"][0][0]["content"]
messages = [
            {"role": "user", "content": sample_data},
]
from unsloth.chat_templates import get_chat_template
FastLanguageModel.for_inference(model)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_value(decoded[0], parse_grid_from_string(test_problems["conversations"][0][1]["content"])))
"""