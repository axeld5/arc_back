PROMPT_V1 = """Task: TRANSDUCTION for ARC. From TRAIN x→y pairs, directly predict TEST.y for TEST.x by pattern imitation.
Think silently. Do NOT explain. Output ONLY the final grid.

Grid encoding:
- Digits 0–9.
- Grid = array of row-strings, e.g. ["012","340"].

INPUT
TRAIN={train_inputs}
TEST={test_input}

Do (silent):
- Infer the simplest transformation consistent with all TRAIN pairs.
- Apply it to TEST.x.
- Validate: shape correct; digits ∈ {0..9}; no extra tokens.

RESPONSE FORMAT
["...","..."]   // the single output grid; no prose, no JSON keys
"""