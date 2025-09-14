def half_list_indices(full_list):
    return len(full_list) // 2

def update_grid_column_based_on_indices(original_grid, column_index):
    num_rows = len(original_grid)
    non_zero_indices = [row_index for row_index in range(num_rows) if original_grid[row_index][column_index]]
    half_index_count = half_list_indices(non_zero_indices)
    for row_index in range(half_index_count):
        original_grid[non_zero_indices[-1 - row_index]][column_index] = 8
    return original_grid

def p(grid, index_range=range):
    num_columns = len(grid[0])
    processed_grid = [row[:] for row in grid]
    for column_index in index_range(num_columns):
        processed_grid = update_grid_column_based_on_indices(processed_grid, column_index)
    return processed_grid
