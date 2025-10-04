import json
import os
import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from typing import *

def array_to_string(arr):
    return str(arr).replace(' ', '')

def format_comparison(output_array, predicted_output):
    expected_str = '\n'.join(array_to_string(row) for row in output_array)
    got_str = '\n'.join(array_to_string(row) for row in predicted_output)
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
        signal.alarm(30)  # 1 minute 30 seconds timeout

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

def inference_loop(model_path: str, base_model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct"):

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

def inference_loop_vllm(model_path: str):
    model = LLM(
        model=model_path,
    )
    sampling = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
    )
    with open("data.json") as f:
        raw = json.load(f)
    total_valid = 0
    for k in range(len(raw["conversations"])):
        print(f"Processing problem {k}")
        sample_data = raw["conversations"][k][0]["content"]
        arrays = raw["arrays"][k]
        prompts = [sample_data]*10
        outputs = model.generate(
            prompts,
            sampling_params=sampling,
        )
        for output in outputs:
            code_resolution = output.outputs[0].text
            print(code_resolution)
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
    sft_merged_save_path = "qwen3_4b_singled_out_sft/final"
    inference_loop(sft_merged_save_path)