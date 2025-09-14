def solve_grid(grid):

    def check_positions(grid, row, col):
        return 3 if grid[row][col] == 0 and grid[row + 5][col] == 0 else 0
    result_grid = []
    for i in range(4):
        result_row = []
        for j in range(4):
            result_row.append(check_positions(grid, i, j))
        result_grid.append(result_row)
    return result_grid
p = solve_grid
