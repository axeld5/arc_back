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

# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

prompt = train_problems["conversations"][0][0]["content"]
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new(),
        ),
        Message.from_role_and_content(Role.USER, prompt),
    ]
)
 
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
 
# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()
 
# --- 2) Run vLLM with prefill ---
llm = LLM(
    model="openai/gpt-oss-20b",
    trust_remote_code=True,
)
 
sampling = SamplingParams(
    max_tokens=4096,
    temperature=1,
    stop_token_ids=stop_token_ids,
)

import time
start_time = time.time()
 
outputs = llm.generate(
    prompt_token_ids=[prefill_ids],   # batch of size 1
    sampling_params=sampling,
)
 
# vLLM gives you both text and token IDs
gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
 
# --- 3) Parse the completion token IDs back into structured Harmony messages ---
entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
 
# 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
for message in entries:
    print(f"{json.dumps(message.to_dict())}")


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

