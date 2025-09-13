import unsloth
import torch
import json
from unsloth import FastLanguageModel
from transduction_test import check_value, parse_grid_from_string
from loader import load_training_problem, list_training_problems

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
print(check_value(decoded[0], parse_grid_from_string(test_problems["conversations"][0][1]["content"])))