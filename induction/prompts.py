PROMPT_V1 = """You are asked to write a program that maps 2D grids to 2D grids.
The mapping is to be inferred from TRAIN and then applied to TEST.
You need to write a python program that takes a list of lists as input and returns a list of lists as output.

Grid encoding:
- Digits 0-9.
- Grid = list of lists of digits.

INPUT
{train_inputs}

Now, output the python program that solves the task. You MUST name your final function "solve".

RESPONSE FORMAT
```python
def solve(input_grid):
    ...
    return output_grid
```
"""