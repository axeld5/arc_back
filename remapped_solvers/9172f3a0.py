def replicate_value(value):
    return [value, value, value]

def replicate_row(row):
    return [replicated_value for element in row for replicated_value in replicate_value(element)]

def replicate_grid(grid):
    return [replicated_row for row in grid for replicated_row in (replicate_row(row),) * 3]

def p(j):
    return replicate_grid(j)
