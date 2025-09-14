def fill_row_with_twos(grid, row_index, num_cols):
    grid[row_index] = [2] * num_cols

def is_col_all_zeros_or_twos(grid, col_index, num_rows):
    return all((grid[row_index][col_index] in [0, 2] for row_index in range(num_rows)))

def fill_col_with_twos(grid, col_index, num_rows):
    for row_index in range(num_rows):
        grid[row_index][col_index] = 2

def p(grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row_index in range(num_rows):
        if sum(grid[row_index]) == 0:
            fill_row_with_twos(grid, row_index, num_cols)
    for col_index in range(num_cols):
        if is_col_all_zeros_or_twos(grid, col_index, num_rows):
            fill_col_with_twos(grid, col_index, num_rows)
    return grid
