import json
from typing import *
from loader import load_training_problem, list_training_problems, load_evaluation_problem, list_evaluation_problems

import unsloth
import os
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

def evaluate_prediction(input_array, output_array, response, debug=False):
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
        local_namespace = {}
        exec(code, local_namespace)
        if 'p' not in local_namespace:
            if debug:
                print(f"Function 'p' not found in generated code")
            return False
        predicted_output = local_namespace['p'](input_array)
        if predicted_output == output_array:
            if debug:
                print(f"✓ Correct prediction for input/output pair")
            return True
        else:
            if debug:
                print(f"✗ Incorrect prediction for input/output pair")
            import difflib
            expected_str = '\n'.join(' '.join(map(str, row)) for row in output_array)
            got_str = '\n'.join(' '.join(map(str, row)) for row in predicted_output)
            diff = '\n'.join(difflib.unified_diff(
                expected_str.splitlines(keepends=True),
                got_str.splitlines(keepends=True),
                fromfile='Expected',
                tofile='Got',
                lineterm=''
            ))
            if debug:
                print(f"Diff:\n{diff}")
            return False
            
    except Exception as e:
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
    return PROMPT_INDUCTION.format(input=input_output_pairs)

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

train_problems = {"conversations":[], "arrays":[]}
data = list_training_problems()
for problem_id in data[:10]:
    print(f"Processing problem {problem_id}")
    problem = load_training_problem(problem_id)
    user_content = {"role":"user", "content":""}
    user_content["content"] = _format_induction_prompt(problem)
    assistant_content = {"role":"assistant", "content":""}
    assistant_content["content"] = _format_code_solution(problem_id)
    train_problems["conversations"].append([user_content, assistant_content])
    train_problems["arrays"].append(problem["train"])

with open('data.json', 'w') as f:
    json.dump(train_problems, f)

from unsloth import FastLanguageModel
base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-Coder-3B-Instruct",
        max_seq_length = 20000,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        fast_inference = True,
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