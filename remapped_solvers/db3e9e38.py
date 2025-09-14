def fill_row_and_column(grid, row, column, value):
    max_columns = len(grid[0])
    if 0 <= column < max_columns:
        grid[row][column] = value

def p(grid, range_func=range):
    num_rows, num_cols = (len(grid), len(grid[0]))
    last_non_zero_row, last_non_zero_col = (0, 0)
    for row in range_func(num_rows):
        for col in range_func(num_cols):
            if grid[row][col]:
                last_non_zero_row, last_non_zero_col = (row + 2, col)
    for offset in range_func(num_cols):
        last_non_zero_row -= 1
        alternating_value = 7 + offset % 2
        for row in range_func(last_non_zero_row):
            fill_row_and_column(grid, row, last_non_zero_col - offset, alternating_value)
            fill_row_and_column(grid, row, last_non_zero_col + offset, alternating_value)
    return grid
