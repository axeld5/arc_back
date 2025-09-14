def p(grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    row_sums = [sum((grid[r][c] != 0 for c in range(num_cols))) for r in range(num_rows)]
    col_sums = [sum((grid[r][c] != 0 for r in range(num_rows))) for c in range(num_cols)]
    max_row_index = max(range(num_rows), key=row_sums.__getitem__)
    max_col_index = max(range(num_cols), key=col_sums.__getitem__)
    max_non_zero = max(row_sums[max_row_index], col_sums[max_col_index])
    non_zero_col_indices = [c for c in range(num_cols) if grid[max_row_index][c]]
    non_zero_row_indices = [r for r in range(num_rows) if grid[r][max_col_index]]
    left_bound, right_bound = (min(non_zero_col_indices), max(non_zero_col_indices))
    top_bound, bottom_bound = (min(non_zero_row_indices), max(non_zero_row_indices))
    pivot_column = [grid[r][max_col_index] for r in range(top_bound, bottom_bound + 1)]

    def find_rotation_offset(sequence):
        first_element = grid[max_row_index][left_bound]
        for i, value in enumerate(sequence):
            if value == first_element:
                offset = (i - left_bound) % max_non_zero
                if all((sequence[(offset + c) % max_non_zero] == grid[max_row_index][c] for c in range(left_bound, right_bound + 1))):
                    return offset
        return None
    forward_offset = find_rotation_offset(pivot_column)
    reverse_offset = find_rotation_offset(pivot_column[::-1])
    should_use_reverse = False
    if forward_offset is None and reverse_offset is not None:
        should_use_reverse = True
        forward_offset = reverse_offset
    if forward_offset is None:
        forward_offset = (max_non_zero - left_bound) % max_non_zero
    solution_grid = [[0] * num_cols for _ in range(num_rows)]
    for r in range(num_rows):
        solution_grid[r][max_col_index] = pivot_column[(r - top_bound) % max_non_zero]
    row_pivot_sequence = pivot_column[::-1] if should_use_reverse else pivot_column
    for c in range(num_cols):
        solution_grid[max_row_index][c] = row_pivot_sequence[(forward_offset + c) % max_non_zero]
    return solution_grid
