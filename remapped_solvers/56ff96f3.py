def find_min_max_positions(grid, target_value):
    row_indices = []
    col_indices = []
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value == target_value:
                row_indices.append(row_index)
                col_indices.append(col_index)
    return (min(row_indices), max(row_indices) + 1, min(col_indices), max(col_indices) + 1)

def fill_grid_with_value(grid, start_row, end_row, start_col, end_col, value):
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            grid[row][col] = value

def p(grid, range_func=range, enumerate_func=enumerate):
    num_rows = len(grid)
    num_cols = len(grid[0])
    result_grid = [[0] * num_cols for _ in range_func(num_rows)]
    unique_values = {value for row in grid for value in row if value != 0}
    for value in unique_values:
        start_row, end_row, start_col, end_col = find_min_max_positions(grid, value)
        fill_grid_with_value(result_grid, start_row, end_row, start_col, end_col, value)
    return result_grid
