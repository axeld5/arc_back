import signal
from loader import load_training_problem, list_training_problems

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
                    return False, output_array
                return False
            predicted_output = local_namespace['p'](input_array)
            signal.alarm(0)
            if predicted_output == output_array:
                if debug:
                    print(f"✓ Correct prediction for input/output pair")
                if get_logs:
                    return True, output_array
                return True
            else:
                if debug:
                    print(f"✗ Incorrect prediction for input/output pair")
                    comparison = format_comparison(output_array, predicted_output)
                    print(f"Comparison (Got -> Expected):\n" + '\n'.join(comparison))
                if get_logs:
                    comparison = format_comparison(output_array, predicted_output)
                    return False, comparison
                return False
        except TimeoutError:
            signal.alarm(0)
            if debug:
                print(f"Code execution timed out after 90 seconds")
            if get_logs:
                return False, output_array
            return False
    except Exception as e:
        signal.alarm(0)
        if debug:
            print(f"Error executing generated code: {e}")
            print(f"Generated code was: {code if 'code' in locals() else 'N/A'}")
        if get_logs:
            return False, output_array
        return False

def get_model():
    pass

def get_program_generation_prompt(model, task):
    pass

def infer_program(model, task):
    pass

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
    for training_array in loaded_training_arrays:
        input_array = training_array["train"][0]["input"]
        output_array = training_array["train"][0]["output"]
        program = extract_program(infer_with_model(model, input_array, output_array))
        library.append(program)
    return library

def make_program_generation_prompt(task_log, primitive):
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

def generate_new_program(model, task_log, primitive):
    pass

def make_round(model, library, gen_num=5, round_num=2):
    tasks = [load_training_problem(problem_id) for problem_id in list_training_problems()]
    solved = []
    for round in range(round_num):
        print(f"Round {round}")
        for task in tasks:
            if task in solved:
                continue
            scores = []
            task_logs = []
            for program in library:
                partial_values = []
                task_log = ""
                for training_array in task:
                    input_array = training_array["train"][0]["input"]
                    output_array = training_array["train"][0]["output"]
                    value, log = evaluate_prediction(input_array, output_array, program, get_logs=True)
                    task_log += "\n" + log
                    if value:
                        partial_values.append(1)
                    else:
                        partial_values.append(0)
                scores.append(sum(partial_values) / len(partial_values))
                task_logs.append(task_log)
            chosen_index = scores.index(max(scores))
            primitive = library[chosen_index]
            task_log = task_logs[chosen_index]
            generated_programs = []
            for _ in range(gen_num):
                programs = generate_new_program(model, task_log, primitive)
                generated_programs.append(programs)
            new_scores = []
            for program in generated_programs:                
                partial_values = []
                for training_array in task:
                    input_array = training_array["train"][0]["input"]
                    output_array = training_array["train"][0]["output"]
                    if evaluate_prediction(input_array, output_array, program):
                        partial_values.append(1)
                    else:
                        partial_values.append(0)
                score = sum(partial_values) / len(partial_values)
                if score == 1:
                    print(f"✓ Solved problem {task}")
                    library.append(program)
                    solved.append(task)
                    break
                new_scores.append(score)
            chosen_index = new_scores.index(max(new_scores))
            library.append(generated_programs[chosen_index])            
    return library, solved

if __name__ == "__main__":
    model = get_model()
    library = []
    library = make_train_library(model, library)
    library, solved = make_round(model, library)
    print(f"Solved {len(solved)} problems")
    print(f"Library size: {len(library)}")
    print(f"Solved: {solved}")
    print(f"Library: {library}")