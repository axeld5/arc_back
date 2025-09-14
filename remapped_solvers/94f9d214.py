def solve_grid(grid):

    def check_condition(row_index, col_index):
        return 2 if grid[row_index][col_index] == 0 == grid[row_index + 4][col_index] else 0
    result_grid = [[check_condition(row, col) for col in range(4)] for row in range(4)]
    return result_grid
p = solve_grid
