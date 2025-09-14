def is_non_zero_or_value_above(grid, row, col):
    return grid[row][col] != 0 or grid[row + 5][col] != 0

def compute_row_result(grid, row):
    return [3 if is_non_zero_or_value_above(grid, row, col) else 0 for col in range(4)]

def p(grid):
    return [compute_row_result(grid, row) for row in range(4)]
