import signal
from loader import load_training_problem, list_training_problems
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
from vllm.inputs import TokensPrompt
from typing import *

def format_comparison(output_array, predicted_output):
    max_rows = max(len(output_array), len(predicted_output) if predicted_output else 0)
    max_cols = 0
    if output_array:
        max_cols = max(max_cols, max([len(row) for row in output_array]))
    if predicted_output:
        max_cols = max(max_cols, max([len(row) if row else 0 for row in predicted_output]))
    comparison = []
    for i in range(max_rows):
        expected_row = output_array[i] if i < len(output_array) else []
        got_row = predicted_output[i] if predicted_output and i < len(predicted_output) else []
        row_comparison = []
        for j in range(max_cols):
            expected_val = expected_row[j] if j < len(expected_row) else None
            got_val = got_row[j] if j < len(got_row) else None
            if expected_val == got_val:
                row_comparison.append(str(got_val) if got_val is not None else "None")
            else:
                row_comparison.append(f"{got_val}->{expected_val}")
        comparison.append(' '.join(row_comparison))
    return comparison

def evaluate_prediction(input_array, output_array, code, get_logs=False, debug=False):
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(90)
        try:
            local_namespace = {}
            exec(code, local_namespace)
            if 'p' not in local_namespace:
                if debug:
                    print(f"Function 'p' not found in generated code")
                signal.alarm(0)
                if get_logs:
                    return False, '\n'.join(format_comparison(output_array, output_array))  
                return False
            predicted_output = local_namespace['p'](input_array)
            signal.alarm(0)
            if predicted_output == output_array:
                if debug:
                    print(f"✓ Correct prediction for input/output pair")
                if get_logs:
                    return True, '\n'.join(format_comparison(output_array, output_array))
                return True
            else:
                if debug:
                    print(f"✗ Incorrect prediction for input/output pair")
                    comparison = format_comparison(output_array, predicted_output)
                    print(f"Comparison (Got -> Expected):\n" + '\n'.join(comparison))
                if get_logs:
                    comparison = format_comparison(output_array, predicted_output)
                    return False, '\n'.join(comparison)
                return False
        except TimeoutError:
            signal.alarm(0)
            if debug:
                print(f"Code execution timed out after 90 seconds")
            if get_logs:
                return False, '\n'.join(format_comparison(output_array, output_array))
            return False
    except Exception as e:
        signal.alarm(0)
        if debug:
            print(f"Error executing generated code: {e}")
            print(f"Generated code was: {code if 'code' in locals() else 'N/A'}")
        if get_logs:
            return False, '\n'.join(format_comparison(output_array, output_array))
        return False

def get_model():
    llm = LLM(
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
    )
    return llm

def grid_to_row_strings(grid: List[List[int]]) -> List[str]:
    return [' '.join(map(str, row)) for row in grid]

def get_input_output_pairs(problem) -> str:
    input_output_pairs = ""
    for i, elem in enumerate(problem['train']):
        pb_input ="\n".join(grid_to_row_strings(elem['input']))
        pb_output = "\n".join(grid_to_row_strings(elem['output']))
        input_output_pairs += f"Input {i+1}:\n{pb_input}\nOutput {i+1}:\n{pb_output}\n\n"
    return input_output_pairs

def get_program_generation_prompt(problem):
    PROMPT_INDUCTION = (
        "Solve the following problem\n\n"
        "Given input/output pairs:\n{io_pairs}\n"
        "Write a python program that solves the problem. Name your final function 'p'.\n"
        "OUTPUT:"
    )
    input_output_pairs = get_input_output_pairs(problem)
    return PROMPT_INDUCTION.format(io_pairs=input_output_pairs)

def convert_to_gptoss_prompt(problem_list, encoding):
    prefill_list = []
    for problem in problem_list:
        prompt = get_program_generation_prompt(problem)
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
    return prefill_list, stop_token_ids

def infer_initial_programs(model, problem_list):
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    prefill_list, stop_token_ids = convert_to_gptoss_prompt(problem_list, encoding)
    prompts = [TokensPrompt(prompt_token_ids=prefill_ids) for prefill_ids in prefill_list]
    sampling = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop_token_ids=stop_token_ids,
    )
    outputs = model.generate(
        prompts,
        sampling_params=sampling,
    )
    response = []
    for output in outputs:
        for gen in output.outputs:
            output_tokens = gen.token_ids
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            for message in entries:
                message = message.to_dict()
                if message["role"] == "assistant":
                    response.append(message["content"][0]["text"])
    return response

def extract_program(response, debug=False):
    start_marker = "```python"
    end_marker = "```"
    start_idx = response.find(start_marker)
    if start_idx == -1:
        if debug:
            print(f"No Python code block found in response")
        return None
    start_idx += len(start_marker)
    end_idx = response.find(end_marker, start_idx)
    if end_idx == -1:
        if debug:
            print(f"No closing code block marker found")
        return None
    code = response[start_idx:end_idx].strip()
    return code

def make_train_library(model, library):
    loaded_training_arrays = [load_training_problem(problem_id) for problem_id in list_training_problems()]
    initial_programs = infer_initial_programs(model, loaded_training_arrays)
    print(len(initial_programs))
    for initial_program in initial_programs:
        program = extract_program(initial_program)
        if program:
            library.append(program)
    print(f"Library size: {len(library)}")
    return library

def make_program_generation_prompt(primitive, task_log):
    return f"""
    You are a program refinement assistant. Your task is to fix a Python program that is not working correctly.

    You are given:
    1. A primitive program that attempts to solve a task but produces incorrect outputs
    2. Test cases showing the input, the program's actual output, and the expected output

    Your goal is to analyze the errors and refine the primitive program to correctly solve all test cases.

    Original program to refine:
    {primitive}

    Test case results (showing where the program fails):
    {task_log}

    Please provide a corrected version of the program that will produce the expected outputs for all test cases. 
    Wrap your solution in ```python``` code blocks and ensure the main function is named 'p'.

    OUTPUT:
    """

def convert_to_proggen_prompt(primitive, task_log, num_gen, encoding):
    prefill_list = []
    for _ in range(num_gen):
        prompt = make_program_generation_prompt(primitive, task_log)
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
    return prefill_list, stop_token_ids

def generate_new_program(model, primitive, task_log, num_gen=5):
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    prefill_list, stop_token_ids = convert_to_proggen_prompt(primitive, task_log, num_gen, encoding)
    prompts = [TokensPrompt(prompt_token_ids=prefill_ids) for prefill_ids in prefill_list]
    sampling = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        stop_token_ids=stop_token_ids,
    )
    outputs = model.generate(
        prompts,
        sampling_params=sampling,
    )
    response = []
    for output in outputs:
        for gen in output.outputs:
            output_tokens = gen.token_ids
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            for message in entries:
                message = message.to_dict()
                if message["role"] == "assistant":
                    response.append(message["content"][0]["text"])
    return response

def list_solved_problems(library, tasks):
    solved = []
    for i,task in enumerate(tasks):
        scores = []
        task_logs = []
        for program in library:
            partial_values = []
            task_log = ""
            for training_array in task["train"]:
                input_array = training_array["input"]
                output_array = training_array["output"]
                value, log = evaluate_prediction(input_array, output_array, program, get_logs=True)
                task_log += "\n" + log
                if value:
                    partial_values.append(1)
                else:
                    partial_values.append(0)
            score = sum(partial_values) / len(partial_values)
            if score == 1:
                print(f"✓ Solved problem {i} with initial library")
                solved.append(task)
    return solved

def make_round(model, tasks, library, solved, gen_num=5, round_num=2):
    for round in range(round_num):
        print(f"Round {round}")
        for i,task in enumerate(tasks):
            if task in solved:
                continue
            scores = []
            task_logs = []
            for program in library:
                partial_values = []
                task_log = ""
                for training_array in task["train"]:
                    input_array = training_array["input"]
                    output_array = training_array["output"]
                    value, log = evaluate_prediction(input_array, output_array, program, get_logs=True)
                    task_log += "\n" + log
                    if value:
                        partial_values.append(1)
                    else:
                        partial_values.append(0)
                score = sum(partial_values) / len(partial_values)
                if score == 1:
                    solved.append(task)
                    print(f"✓ Solved problem {i} with additional program")
                    break
                scores.append(score)
                task_logs.append(task_log)
            chosen_index = scores.index(max(scores))
            primitive = library[chosen_index]
            task_log = task_logs[chosen_index]
            generated_programs = generate_new_program(model, primitive, task_log, gen_num)
            new_programs = []
            for program in generated_programs:
                program = extract_program(program)
                if program:
                    new_programs.append(program)
            new_scores = []
            for program in new_programs:                
                partial_values = []
                for training_array in task["train"]:
                    input_array = training_array["input"]
                    output_array = training_array["output"]
                    if evaluate_prediction(input_array, output_array, program):
                        partial_values.append(1)
                    else:
                        partial_values.append(0)
                score = sum(partial_values) / len(partial_values)
                if score == 1:
                    print(f"✓ Solved problem {i}")
                    library.append(program)
                    solved.append(task)
                    break
                new_scores.append(score)
            chosen_index = new_scores.index(max(new_scores))
            library.append(generated_programs[chosen_index])            
    return library, solved

if __name__ == "__main__":
    model = get_model()
    tasks = [load_training_problem(problem_id) for problem_id in list_training_problems()]
    library = []
    library = make_train_library(model, library)
    solved = list_solved_problems(library, tasks)
    library, solved = make_round(model, tasks, library, solved)
    solved = list_solved_problems(library, tasks)
    print(f"Solved {len(solved)} problems")
    print(f"Library size: {len(library)}")
    #print(f"Solved: {solved}")