def apply_value_to_diagonal(grid, result_grid, row, col, value):
    max_steps = 6 - max(row, col)
    for i in range(max_steps):
        result_grid[row + i][col + i] = value

def solve_grid_values(grid):
    result_grid = [[0] * 6 for _ in range(6)]
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value != 0:
                apply_value_to_diagonal(grid, result_grid, row_index, col_index, value)
    return result_grid

def p(g, e=enumerate):
    return solve_grid_values(g)
