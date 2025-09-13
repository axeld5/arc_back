import unsloth
import torch
import json
from unsloth import FastLanguageModel
from loader import load_training_problem, list_training_problems
from typing import *
import re

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


data = list_training_problems()
problem_id = data[0]
example = load_training_problem(problem_id)

print(example)

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

with open("test_problems.json") as f:
        raw = json.load(f)
print(raw)
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
print(check_value(decoded[0], parse_grid_from_string(test_problems["conversations"][0][1]["content"])))