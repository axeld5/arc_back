def combine_columns(grid, row, col):
    return grid[row][col] or grid[row + 6][col]

def combine_rows(grid, row):
    num_cols = len(grid[0])
    return [combine_columns(grid, row, col) for col in range(num_cols)]

def p(grid):
    num_rows_to_process = 5
    return [combine_rows(grid, row) for row in range(num_rows_to_process)]
