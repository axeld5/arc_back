import json
from typing import Optional
from loader import load_training_problem, list_training_problems, load_evaluation_problem, list_evaluation_problems

PROMPT_INDUCTION = (
    "Solve the following problem\n\n"
    "Given input/output pairs:\n{io_pairs}\n"
    "Write a python program that solves the problem. Name your final function 'p'.\n"
    "OUTPUT:"
)

def array_to_string(arr):
    return str(arr).replace(' ', '')

def _format_induction_prompt(problem) -> str:
    input_output_pairs = ""
    for i, elem in enumerate(problem['train']):
        pb_input = array_to_string(elem['input'])
        pb_output = array_to_string(elem['output'])
        input_output_pairs += f"<input_{i+1}> {pb_input} </input_{i+1}> <output_{i+1}> {pb_output} </output_{i+1}>"
    return PROMPT_INDUCTION.format(io_pairs=input_output_pairs)

def _format_code_solution(problem_id):
    reasoning_path = f"reasoning_files/{problem_id}.txt"
    with open(reasoning_path, 'r', encoding='utf-8') as f:
        reasoning = f.read()
    solver_path = f"remapped_solvers/{problem_id}.py"
    with open(solver_path, 'r', encoding='utf-8') as f:
        solver_code = f.read()
    solution = f"""
    ```python
    {solver_code}
    ```"""
    return reasoning, solution

def get_data(max_samples: Optional[int] = None):
    train_problems = {"conversations":[], "arrays":[]}
    data = list_training_problems()
    if max_samples is None:
        max_samples = len(data)
    for problem_id in data[:max_samples]:
        print(f"Processing problem {problem_id}")
        problem = load_training_problem(problem_id)
        user_content = {"role":"user", "content":"", "thinking":""}
        user_content["content"] = _format_induction_prompt(problem)
        assistant_content = {"role":"assistant", "content":"", "thinking":""}
        reasoning, solution = _format_code_solution(problem_id)
        assistant_content["thinking"] = reasoning
        assistant_content["content"] = solution
        train_problems["conversations"].append([user_content, assistant_content])
        train_problems["arrays"].append(problem["train"])

    test_problems = {"conversations":[], "arrays":[]}
    eval_data = list_evaluation_problems()
    for problem_id in eval_data:
        problem = load_evaluation_problem(problem_id)
        user_content = {"role":"user", "content":"", "thinking":""}
        user_content["content"] = _format_induction_prompt(problem)
        test_problems["conversations"].append([user_content])
        test_problems["arrays"].append(problem["train"])

    with open('data.json', 'w') as f:
        json.dump(train_problems, f)
    with open('test_problems.json', 'w') as f:
        json.dump(test_problems, f)


if __name__ == "__main__":
    get_data(max_samples=10)

