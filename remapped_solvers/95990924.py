def solve_grid(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))

    def is_all_fives(row, col):
        return grid[row][col] == grid[row][col + 1] == grid[row + 1][col] == grid[row + 1][col + 1] == 5

    def modify_adjacent_cells(row, col):
        if row > 0 and col > 0:
            grid[row - 1][col - 1] = 1
        if row > 0 and col + 2 < num_cols:
            grid[row - 1][col + 2] = 2
        if row + 2 < num_rows and col > 0:
            grid[row + 2][col - 1] = 3
        if row + 2 < num_rows and col + 2 < num_cols:
            grid[row + 2][col + 2] = 4
    for row in range(num_rows - 1):
        for col in range(num_cols - 1):
            if is_all_fives(row, col):
                modify_adjacent_cells(row, col)
    return grid
p = solve_grid
