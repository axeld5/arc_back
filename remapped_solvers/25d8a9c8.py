def is_uniform_row(row):
    return len(set(row)) == 1

def transform_row(row):
    return [5] * 3 if is_uniform_row(row) else [0] * 3

def p(grid):
    return [transform_row(row) for row in grid]
