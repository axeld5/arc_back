def repeat_pattern_across_columns(grid, pattern_row_index, num_rows, num_cols):
    pattern = grid[pattern_row_index] * (num_cols + pattern_row_index)
    for row_index in range(2, num_rows):
        grid[row_index] = [pattern[row_index - 2] for _ in range(num_cols)]
    return grid

def p(grid, row_range=range):
    num_rows = len(grid)
    num_cols = len(grid[0])
    modified_grid = repeat_pattern_across_columns(grid, 0, num_rows, num_cols)
    return modified_grid
