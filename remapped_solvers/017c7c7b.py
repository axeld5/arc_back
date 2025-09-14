def double_elements(grid):
    return [[element * 2 for element in row] for row in grid]

def need_special_handling(grid):
    return grid[1] != grid[4]

def extend_grid(grid):
    if need_special_handling(grid):
        return grid + grid[2:5]
    else:
        return grid + grid[:3]

def p(grid):
    extended_grid = extend_grid(grid)
    result = double_elements(extended_grid)
    return result
