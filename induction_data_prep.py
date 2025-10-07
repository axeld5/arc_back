import json
from typing import Optional
from loader import load_training_problem, list_training_problems, load_evaluation_problem, list_evaluation_problems

PROMPT_INDUCTION = (
    """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by
    generating Python code.
    Your goal is to analyze input-output grid pairs. The outputs were produced by applying a
    transformation rule to the inputs. Implement the transformation rules as a Python function.
    You should only write the implemented the transformation in code.
    You must write code in triple backticks (```python and then ```). You must write a function
    called ```transform``` which takes a single argument, the input grid as ```list[list[int]]```, and
    returns the transformed grid (also as ```list[list[int]]```).
    You should make sure that you implement a version of the transformation which works in general
    (at least for all given input-output pairs and test input pairs).
    The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Red; 3:
    Green; 4:Yellow; 5:Grey; 6:Pink; 7:Orange; 8:Purple; 9:Brown
    Now, solve the following ARC-AGI task:
    # Task to solve
    {io_pairs}"""
)

def array_to_string(arr):
    return str(arr).replace(' ', '')

def format_grid_display(grid):
    """Format a grid in a nice 2D display format."""
    import numpy as np
    arr = np.array(grid)
    height, width = arr.shape
    
    # Format the grid with brackets
    lines = []
    for i, row in enumerate(arr):
        row_str = " ".join(str(x) for x in row)
        if i == 0:
            lines.append(f"[[{row_str}]")
        elif i == len(arr) - 1:
            lines.append(f" [{row_str}]]")
        else:
            lines.append(f" [{row_str}]")
    
    return f"(grid shape: {height} by {width}):\n" + "\n".join(lines)

def _format_induction_prompt(problem) -> str:
    input_output_pairs = ""
    for i, elem in enumerate(problem['train']):
        pb_input = format_grid_display(elem['input'])
        pb_output = format_grid_display(elem['output'])
        input_output_pairs += f"## Input {i+1} {pb_input}\n## Output {i+1} {pb_output}\n"
    return PROMPT_INDUCTION.format(io_pairs=input_output_pairs)

def _format_code_solution(problem_id):
    reasoning_path = f"reasoning_files/{problem_id}.txt"
    with open(reasoning_path, 'r', encoding='utf-8') as f:
        reasoning = f.read()
    solver_path = f"remapped_solvers/{problem_id}.py"
    with open(solver_path, 'r', encoding='utf-8') as f:
        solver_code = f.read()
    modified_solver_code = solver_code.replace("def p(", "def transform(").replace("p = solve", "transform = solve").replace("p = lambda", "transform = lambda")
    modified_solver_code = modified_solver_code.replace("p = find_largest_subrectangle_with_most_twos", "transform = find_largest_subrectangle_with_most_twos").replace("p = process_grid", "transform = process_grid")
    #solution = f"""<think>
    #{reasoning}
    #</think><
    #Here's the code that solves the problem:
    solution = f"""
    ```python
    {modified_solver_code}
    ```"""
    return solution

def test_all_problems():
    """Test that all remapped_solvers transform functions solve their respective problems."""
    data = list_training_problems()
    total_problems = len(data)
    passed = 0
    failed = []
    
    for problem_id in data:
        print(f"Testing problem {problem_id}...")
        problem = load_training_problem(problem_id)
        
        # Load the solver code
        solver_path = f"remapped_solvers/{problem_id}.py"
        try:
            with open(solver_path, 'r', encoding='utf-8') as f:
                solver_code = f.read()
            
            # Replace p with transform to match expected format
            modified_solver_code = solver_code.replace("def p(", "def transform(").replace("p = solve", "transform = solve").replace("p = lambda", "transform = lambda")
            modified_solver_code = modified_solver_code.replace("p = find_largest_subrectangle_with_most_twos", "transform = find_largest_subrectangle_with_most_twos").replace("p = process_grid", "transform = process_grid")
            # Execute the solver code
            local_namespace = {}
            exec(modified_solver_code, local_namespace)
            
            if 'transform' not in local_namespace:
                print(f"  ✗ No transform function found in {solver_path}")
                failed.append((problem_id, "No transform function"))
                continue
            
            transform_func = local_namespace['transform']
            
            # Test on all training examples
            all_correct = True
            for i, example in enumerate(problem['train']):
                input_array = example['input']
                expected_output = example['output']
                
                try:
                    predicted_output = transform_func(input_array)
                    if predicted_output != expected_output:
                        print(f"  ✗ Example {i+1} failed")
                        all_correct = False
                        break
                except Exception as e:
                    print(f"  ✗ Example {i+1} raised exception: {e}")
                    all_correct = False
                    break
            
            if all_correct:
                print(f"  ✓ All {len(problem['train'])} examples passed")
                passed += 1
            else:
                failed.append((problem_id, "Incorrect output"))
                
        except FileNotFoundError:
            print(f"  ✗ Solver file not found: {solver_path}")
            failed.append((problem_id, "File not found"))
        except Exception as e:
            print(f"  ✗ Error processing {problem_id}: {e}")
            failed.append((problem_id, str(e)))
    
    print("\n" + "="*80)
    print(f"Test Results: {passed}/{total_problems} problems passed")
    print("="*80)
    
    if failed:
        print("\nFailed problems:")
        for problem_id, reason in failed:
            print(f"  - {problem_id}: {reason}")
    
    return passed == total_problems


def get_data(max_samples: Optional[int] = None):
    train_problems = {"conversations":[], "arrays":[]}
    data = list_training_problems()
    if max_samples is None:
        max_samples = len(data)
    for problem_id in data[:max_samples]:
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

    full_set = {"conversations":train_problems["conversations"] + test_problems["conversations"], "arrays":train_problems["arrays"] + test_problems["arrays"]}

    with open('data.json', 'w') as f:
        json.dump(train_problems, f)
    with open('test_problems.json', 'w') as f:
        json.dump(test_problems, f)
    with open('full_set.json', 'w') as f:
        json.dump(full_set, f)


if __name__ == "__main__":
    # Uncomment the function you want to run:
    get_data(max_samples=None)
    #test_all_problems()