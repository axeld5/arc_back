import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
 
from vllm import LLM, SamplingParams
from loader import load_training_problem, list_training_problems
from typing import *

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

def main():
    train_problems = {"conversations": [], "arrays": []}
    data = list_training_problems()
    for problem_id in data[:100]:
        print(f"Processing problem {problem_id}")
        problem = load_training_problem(problem_id)
        user_content = {"role": "user", "content": _format_induction_prompt(problem)}
        assistant_content = {"role": "assistant", "content": _format_code_solution(problem_id)}
        train_problems["conversations"].append([user_content, assistant_content])
        train_problems["arrays"].append(problem["train"])

    # --- 1) Render the prefill with Harmony ---
    prefill_list = []
    for problem in train_problems["conversations"]:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        prompt = problem[0]["content"]
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(Role.DEVELOPER, DeveloperContent.new()),
                Message.from_role_and_content(Role.USER, prompt),
            ]
        )
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        prefill_list.append(prefill_ids)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # --- 2) Run vLLM with prefill ---
    sampling = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop_token_ids=stop_token_ids,
    )

    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
    )

    import time
    start_time = time.time()
    from vllm.inputs import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=prefill_ids) for prefill_ids in prefill_list]

    outputs = llm.generate(
        prompts,  # batch size 1
        sampling_params=sampling,
    )

    gen = outputs[0].outputs[0]
    text = gen.text
    output_tokens = gen.token_ids

    # --- 3) Parse back to Harmony entries ---
    entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
    for message in entries:
        message = message.to_dict()
        if message["role"] == "assistant":
            print(message["content"][0]["text"])

    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    # vLLM will also set this, but doing it here is fine and explicit.
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set – safe to ignore
        pass
    main()