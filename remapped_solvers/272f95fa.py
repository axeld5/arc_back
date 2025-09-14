def solve_grid(grid):

    def is_row_all_eights(row_idx):
        return all((value == 8 for value in grid[row_idx]))

    def is_column_all_eights(col_idx):
        return all((grid[row_idx][col_idx] == 8 for row_idx in range(num_rows)))

    def determine_value(row_idx, col_idx):
        if row_idx < min_row and min_col < col_idx < max_col:
            return 2
        elif min_row < row_idx < max_row and col_idx < min_col:
            return 4
        elif min_row < row_idx < max_row and min_col < col_idx < max_col:
            return 6
        elif min_row < row_idx < max_row and col_idx > max_col:
            return 3
        elif row_idx > max_row and min_col < col_idx < max_col:
            return 1
        return 0
    num_rows, num_cols = (len(grid), len(grid[0]))
    result_grid = [row[:] for row in grid]
    min_row, max_row = ([row_idx for row_idx in range(num_rows) if is_row_all_eights(row_idx)][0], [row_idx for row_idx in range(num_rows) if is_row_all_eights(row_idx)][-1])
    min_col, max_col = ([col_idx for col_idx in range(num_cols) if is_column_all_eights(col_idx)][0], [col_idx for col_idx in range(num_cols) if is_column_all_eights(col_idx)][-1])
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if result_grid[row_idx][col_idx] == 0:
                result_grid[row_idx][col_idx] = determine_value(row_idx, col_idx)
    return result_grid
p = solve_grid
