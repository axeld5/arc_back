def apply_initial_row_to_column(grid, col_index, num_rows):
    for row_index in range(num_rows):
        if row_index % 2 == 0:
            grid[row_index][col_index] = grid[0][col_index]
        else:
            if col_index > 0:
                grid[row_index][col_index - 1] = grid[0][col_index]
            if col_index < len(grid[0]) - 1:
                grid[row_index][col_index + 1] = grid[0][col_index]

def p(original_grid, range_function=range):
    transformed_grid = [row[:] for row in original_grid]
    num_rows, num_cols = (len(original_grid), len(original_grid[0]))
    for col_index in range_function(num_cols):
        if original_grid[0][col_index] != 0:
            apply_initial_row_to_column(transformed_grid, col_index, num_rows)
    return transformed_grid
