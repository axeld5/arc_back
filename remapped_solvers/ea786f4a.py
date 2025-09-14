def zero_diagonals(grid):
    size = len(grid)
    for index in range(size):
        grid[index][index] = 0
        grid[index][size - index - 1] = 0
    return grid

def p(grid):
    return zero_diagonals(grid)
