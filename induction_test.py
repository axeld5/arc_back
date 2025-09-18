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
            print(f"Diff:\n{diff}")
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
for problem_id in data[:10]:
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

def run_sft(
    dataset_path: str,
    output_dir: str = "qwen3_4b_singled_out_sft",
    base_model: str = "unsloth/Qwen3-8B",
    learning_rate: float = 8e-5,
    num_train_epochs: int = 5,
    use_compile: bool = False,
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
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
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
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
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
    model.save_pretrained(os.path.join(output_dir, "final"))
    model.save_pretrained_merged(os.path.join(output_dir, "vllm"), tokenizer, save_method = "merged_16bit",)
    try:
        tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    except Exception:
        pass    
    return os.path.join(output_dir, "final")

#run_sft("data.json")

from unsloth import FastLanguageModel
"""base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-8B",
        max_seq_length = 20000,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "qwen3_4b_singled_out_sft/final")
FastLanguageModel.for_inference(model)"""

from vllm import LLM, SamplingParams
import torch
model_id = "Qwen/Qwen3-4B-Instruct-2507"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    #quantization="bitsandbytes"
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=5000)

with open("data.json") as f:
    raw = json.load(f)

total_valid = 0
for k in range(len(raw["conversations"])):
    print(f"Processing problem {k}")
    sample_data = raw["conversations"][k][0]["content"]
    #messages = [
    #    {"role": "user", "content": sample_data},
    #]
    prompts = [sample_data]*10
    arrays = raw["arrays"][k]
    """inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")"""
    decoded = llm.generate(prompts, sampling_params)
    for p in range(10):
        #outputs = model.generate(input_ids = inputs, max_new_tokens = 5000,  
        #temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, use_cache = True)
        #generated_tokens = outputs[:, inputs.shape[-1]:]
        #decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        print(decoded[p])
        cnt = 0
        for inp_out in arrays:
            input_array = inp_out["input"]
            output_array = inp_out["output"]
            if evaluate_prediction(input_array, output_array, decoded[0]):
                cnt += 1
        if cnt == len(arrays):
            print(f"✓ problem {k}")
            total_valid += 1
            break
        else:
            print(f"✗ problem {k}")
print(f"Total valid: {total_valid}/{len(raw['conversations'])}")

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
        # Expecting [ {"role":"user"}, {"role":"assistant"} ]
        user_msg = convo[0]["content"]
        result.append({
            "prompt": [
                {"role": "user", "content": user_msg}
            ],
            "arrays": array_list
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
    max_seq_length = 20000 # Can increase for longer reasoning traces
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
            evaluate_code_validity
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
        print(decoded[0])
        cnt = 0
        for inp_out in arrays:
            input_array = inp_out["input"]
            output_array = inp_out["output"]
            if evaluate_prediction(input_array, output_array, decoded[0]):
                cnt += 1
        if cnt == len(arrays):
            print(f"✓ problem {k}")
            total_valid += 1
            break
        else:
            print(f"✗ problem {k}")
print(f"Total valid: {total_valid}/{len(raw['conversations'])}")