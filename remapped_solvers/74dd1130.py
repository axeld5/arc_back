def transpose_grid(grid):
    return [list(row) for row in zip(*grid)]

def p(grid):
    return transpose_grid(grid)
