from copy import deepcopy

def get_mirrored_index(i, j):
    return (9 - i, 9 - j)

def transform_grid(grid, offset, multiplier):
    transformed_grid = deepcopy(grid)
    for i in range(10 - offset):
        for j in range(5):
            row, col = (i + offset, j + offset)
            if not grid[row][col]:
                mirrored_row, mirrored_col = get_mirrored_index(i, j)
                transformed_grid[row][col] += multiplier * grid[mirrored_row][mirrored_col]
    return transformed_grid

def is_symmetric(grid, offset):
    for i in range(10 - offset):
        for j in range(5):
            row, col = (i + offset, j + offset)
            mirrored_row, mirrored_col = get_mirrored_index(i, j)
            if grid[row][col] != grid[mirrored_row][mirrored_col]:
                return False
    return True

def p(grid):
    transformed_once = transform_grid(grid, 1, 1)
    if is_symmetric(transformed_once, 1):
        return transform_grid(grid, 1, 2)
    else:
        return transform_grid(grid, 0, 2)
