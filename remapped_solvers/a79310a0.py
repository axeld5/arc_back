def rotate_grid_right(grid):
    return [grid[-1]] + grid[:-1]

def replace_eights_with_twos(grid):
    return [[2 if cell == 8 else cell for cell in row] for row in grid]

def p(grid):
    rotated_grid = rotate_grid_right(grid)
    transformed_grid = replace_eights_with_twos(rotated_grid)
    return transformed_grid
