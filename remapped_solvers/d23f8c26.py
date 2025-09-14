def copy_middle_column(grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    middle_col_index = num_cols // 2
    new_grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    for row_index in range(num_rows):
        new_grid[row_index][middle_col_index] = grid[row_index][middle_col_index]
    return new_grid

def p(grid):
    return copy_middle_column(grid)
