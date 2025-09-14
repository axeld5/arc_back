def find_non_zero_indices(grid):
    non_zero_indices = [index for row in grid for index, value in enumerate(row) if value > 0]
    start_index = min(non_zero_indices)
    end_index = max(non_zero_indices) + 1
    return (start_index, end_index)

def process_grid(grid, start_index, end_index):
    processed_rows = []
    for row in grid:
        if max(row) > 0:
            processed_row_segment = row[start_index:end_index]
            processed_rows.append(processed_row_segment * 2)
    return processed_rows

def p(grid):
    start_index, end_index = find_non_zero_indices(grid)
    return process_grid(grid, start_index, end_index)
