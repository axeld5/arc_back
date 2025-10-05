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
        # Find all python code blocks
        code_blocks = []
        search_pos = 0
        while True:
            start_idx = response.find(start_marker, search_pos)
            if start_idx == -1:
                break
            start_idx += len(start_marker)
            end_idx = response.find(end_marker, start_idx)
            if end_idx == -1:
                break
            code_blocks.append(response[start_idx:end_idx].strip())
            search_pos = end_idx + len(end_marker)
        if not code_blocks:
            if debug:
                print(f"No Python code block found in response")
            return False
        code = code_blocks[-1]

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
            #print(f"Generated code was: {code if 'code' in locals() else 'N/A'}")
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

def inference_loop_vllm(model_path: str, attempts_per_problem: int = 10):
    from transformers import AutoTokenizer
    
    # Load tokenizer and model once
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLM(model=model_path, trust_remote_code=True)
    sampling = SamplingParams(max_tokens=4096, temperature=1.0)
    
    with open("test_problems.json") as f:
        raw = json.load(f)
    
    # Prepare ALL prompts upfront (outside loop)
    all_prompts = []
    problem_indices = []  # Track which problem each prompt belongs to
    
    for k in range(len(raw["conversations"])):
        sample_data = raw["conversations"][k][0]["content"]
        messages = [{"role": "user", "content": sample_data}]
        
        # Apply chat template once per problem
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create multiple attempts per problem
        for _ in range(attempts_per_problem):
            all_prompts.append(formatted_prompt)
            problem_indices.append(k)
    
    print(f"Generating {len(all_prompts)} outputs in batch ({attempts_per_problem} attempts × {len(raw['conversations'])} problems)...")
    
    # Generate ALL outputs in ONE batch call - this is where vLLM shines!
    outputs = model.generate(
        all_prompts,
        sampling_params=sampling,
    )
    
    print("Batch generation complete! Evaluating results...")
    
    # Process results
    total_valid = 0
    for k in range(len(raw["conversations"])):
        print(f"Processing problem {k}")
        arrays = raw["arrays"][k]
        
        # Get all outputs for this problem
        problem_outputs = [
            outputs[i] for i, prob_idx in enumerate(problem_indices) 
            if prob_idx == k
        ]
        
        # Try each attempt
        found_solution = False
        for i, output in enumerate(problem_outputs):
            code_resolution = output.outputs[0].text            
            cnt = 0
            for inp_out in arrays:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                if evaluate_prediction(input_array, output_array, code_resolution, debug=(i==0)):
                    cnt += 1
            
            if cnt == len(arrays):
                print(f"✓ problem {k} (solved on attempt {i+1}/{attempts_per_problem})")
                total_valid += 1
                found_solution = True
                break
        
        if not found_solution:
            print(f"✗ problem {k} (failed all {attempts_per_problem} attempts)")
    
    print(f"\nTotal valid: {total_valid}/{len(raw['conversations'])}")

def inference_loop_vllm_gptoss(model_name: str = "openai/gpt-oss-20b", attempts_per_problem: int = 10):
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )
    from vllm.inputs import TokensPrompt
    
    # Load Harmony encoding for gpt-oss
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    
    # Load model
    model = LLM(
        model=model_name,
        trust_remote_code=True,
    )
    
    sampling = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop_token_ids=stop_token_ids,
    )
    
    with open("data.json") as f:
        raw = json.load(f)
    
    # Prepare ALL prompts upfront (outside loop)
    all_token_prompts = []
    problem_indices = []  # Track which problem each prompt belongs to
    
    print("Tokenizing all prompts with Harmony encoding...")
    for k in range(len(raw["conversations"])):
        sample_data = raw["conversations"][k][0]["content"]
        
        # Convert to Harmony conversation format
        convo = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new()),
            Message.from_role_and_content(Role.USER, sample_data),
        ])
        
        # Render to token IDs
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        
        # Create multiple attempts per problem
        for _ in range(attempts_per_problem):
            all_token_prompts.append(TokensPrompt(prompt_token_ids=prefill_ids))
            problem_indices.append(k)
    
    print(f"Generating {len(all_token_prompts)} outputs in batch ({attempts_per_problem} attempts × {len(raw['conversations'])} problems)...")
    
    # Generate ALL outputs in ONE batch call
    outputs = model.generate(
        all_token_prompts,
        sampling_params=sampling,
    )
    
    print("Batch generation complete! Parsing with Harmony encoding...")
    
    # Parse all responses using Harmony encoding
    all_responses = []
    for output in outputs:
        gen = output.outputs[0]
        output_tokens = gen.token_ids
        entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
        
        # Extract text from assistant messages
        response_text = ""
        for message in entries:
            message_dict = message.to_dict()
            if message_dict["role"] == "assistant":
                # Content is a list of content items
                for content_item in message_dict["content"]:
                    if "text" in content_item:
                        response_text += content_item["text"]
        all_responses.append(response_text)
    
    print("Evaluating results...")
    
    # Process results
    total_valid = 0
    for k in range(len(raw["conversations"])):
        print(f"Processing problem {k}")
        arrays = raw["arrays"][k]
        
        # Get all responses for this problem
        problem_responses = [
            all_responses[i] for i, prob_idx in enumerate(problem_indices) 
            if prob_idx == k
        ]
        
        # Try each attempt
        found_solution = False
        for i, code_resolution in enumerate(problem_responses):            
            cnt = 0
            for inp_out in arrays:
                input_array = inp_out["input"]
                output_array = inp_out["output"]
                if evaluate_prediction(input_array, output_array, code_resolution, debug=(i==0)):
                    cnt += 1
            
            if cnt == len(arrays):
                print(f"✓ problem {k} (solved on attempt {i+1}/{attempts_per_problem})")
                total_valid += 1
                found_solution = True
                break
        
        if not found_solution:
            print(f"✗ problem {k} (failed all {attempts_per_problem} attempts)")
    
    print(f"\nTotal valid: {total_valid}/{len(raw['conversations'])}")

if __name__ == "__main__":
    sft_merged_save_path = "qwen2.5_7b_singled_out_sft/merged"
    #inference_loop_vllm("unsloth/Qwen2.5-Coder-7B-Instruct")
    inference_loop_vllm(sft_merged_save_path)
    rl_merged_save_path = "qwen2.5_7b_singled_out_rl_partial/merged"
    inference_loop_vllm(rl_merged_save_path)
    rl_merged_save_path = "qwen2.5_7b_singled_out_rl_full/merged"
    inference_loop_vllm(rl_merged_save_path)
    
    # Choose which inference to run:
    # For Qwen models (SFT/RL fine-tuned):
    
    # For gpt-oss models:
    #inference_loop_vllm_gptoss(model_name="gptoss_induction_sft/merged")