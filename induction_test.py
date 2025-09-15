import json
from typing import *
from loader import load_training_problem, list_training_problems, load_evaluation_problem, list_evaluation_problems


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

def evaluate_prediction(input_array, output_array, response):
    try:
        start_marker = "```python"
        end_marker = "```"
        start_idx = response.find(start_marker)
        if start_idx == -1:
            print(f"No Python code block found in response")
            return False
        start_idx += len(start_marker)
        end_idx = response.find(end_marker, start_idx)
        if end_idx == -1:
            print(f"No closing code block marker found")
            return False
        code = response[start_idx:end_idx].strip()
        local_namespace = {}
        exec(code, local_namespace)
        if 'p' not in local_namespace:
            print(f"Function 'p' not found in generated code")
            return False
        predicted_output = local_namespace['p'](input_array)
        if predicted_output == output_array:
            print(f"✓ Correct prediction for input/output pair")
            return True
        else:
            print(f"✗ Incorrect prediction for input/output pair")
            print(f"Expected: {output_array}")
            print(f"Got: {predicted_output}")
            return False
            
    except Exception as e:
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
for problem_id in data:
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
    base_model: str = "unsloth/Qwen3-4B-Instruct-2507",
    learning_rate: float = 8e-5,
    num_train_epochs: int = 10,
    use_compile: bool = False,
):      

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,                 # use the arg instead of hardcoding
        max_seq_length = 20000,
        dtype = compute_dtype,
        load_in_4bit = True,
        attn_implementation = "sdpa",   # ← key change
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
        per_device_train_batch_size = 4,
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
        max_seq_length=20000,
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


model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen3_4b_singled_out_sft/final", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = 20000,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        fast_inference = True
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
outputs = model.generate(input_ids = inputs, max_new_tokens = 5000, use_cache = True)
generated_tokens = outputs[:, inputs.shape[-1]:]
decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(decoded[0])

arrays = raw["arrays"][0]

for inp_out in arrays:
    input_array = inp_out["input"]
    output_array = inp_out["output"]
    evaluate_prediction(input_array, output_array, decoded[0])
