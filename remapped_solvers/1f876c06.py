def solve_diagonal_pattern(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))
    pattern_map = {}
    result_grid = [row[:] for row in grid]
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            value = grid[row_index][col_index]
            if value:
                pattern_map.setdefault(value, []).append((row_index, col_index))
    for value in pattern_map:
        start, end = pattern_map[value]
        start_row, start_col = start
        end_row, end_col = end
        row_step = 1 if end_row > start_row else -1
        col_step = 1 if end_col > start_col else -1
        for step in range(abs(end_row - start_row) + 1):
            result_row = start_row + step * row_step
            result_col = start_col + step * col_step
            result_grid[result_row][result_col] = value
    return result_grid

def p(j, A=range):
    return solve_diagonal_pattern(j)
