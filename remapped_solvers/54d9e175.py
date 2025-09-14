def apply_transformation(grid, row, col, increment_value):
    subgrid_size = 3
    for i in range(subgrid_size):
        for j in range(subgrid_size):
            grid[row - 1 + i][col - 1 + j] = increment_value

def transform_grid(grid):
    modified_grid = [row[:] for row in grid]
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row in range(1, num_rows, 4):
        for col in range(1, num_cols, 4):
            increment_value = grid[row][col] + 5
            apply_transformation(modified_grid, row, col, increment_value)
    return modified_grid

def p(grid):
    return transform_grid(grid)
