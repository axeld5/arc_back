PROMPT_V1 = """Task: TRANSDUCTION for ARC. From TRAIN INPUT/OUTPUT grid pairs, directly predict TEST GRID OUTPUT for TEST GRID INPUT by pattern imitation.
Think silently. Do NOT explain. Output ONLY the final grid, correcting the test output placeholder.

Grid encoding:
- Digits 0-9.
- Grid rows separated by semicolons, e.g. "012;340"

{train_pairs}

TEST INPUT: {test_input}
TEST OUTPUT PLACEHOLDER: {test_output_placeholder}
TEST OUTPUT: """