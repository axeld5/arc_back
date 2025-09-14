import json
from typing import *
from loader import load_training_problem, list_training_problems

import re
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

def grid_to_row_strings(grid: List[List[int]]) -> List[str]:
    return [' '.join(map(str, row)) for row in grid]

def _format_single_prompt(input_grid: List[List[int]], placeholder_rows: str, task_id: str) -> str:
    input_str = "\n".join(grid_to_row_strings(input_grid))
    return PROMPT_V2.format(task_id=task_id, input=input_str, placeholder=placeholder_rows)

PROMPT_V2 = (
    "Solve task {task_id}\n\n"
    "INPUT:\n{input}\n"
    "OUTPUT PLACEHOLDER:\n{placeholder}\n"
    "OUTPUT:"
)

import numpy as np

problem_id = "c909285e"
problem = load_training_problem(problem_id)
sample_data = problem["test"][0]["input"]
content = _format_single_prompt(sample_data, "\n".join(grid_to_row_strings(np.zeros((3, 3)).astype(int))), problem_id)
messages = [
            {"role": "user", "content": content},
]
print(messages)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen3_4b_singled_out_sft/final", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 8192, use_cache = True)
print(outputs)
"""
outputs = model.generate(input_ids = inputs, max_new_tokens = 4096, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tok.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])
print(check_value(decoded[0], problem["test"][0]["output"]))
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
print(check_value(decoded[0], problem["test"][0]["output"]))
for _ in range(16):
    content = _format_single_prompt(sample_data, decoded[0], problem_id)
    messages = [
                {"role": "user", "content": content},
    ]
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
    print(check_value(decoded[0], problem["test"][0]["output"]))"""