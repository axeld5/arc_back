def p(grid):

    def create_initial_snapshot(grid):
        return [[grid[row][col] for col in range(3)] for row in range(3)]

    def update_surrounding(grid, snapshot, row, col):
        for delta_row in range(-1, 2):
            for delta_col in range(-1, 2):
                new_row = row + delta_row
                new_col = col + delta_col
                if 0 <= new_row < 9 and 4 <= new_col < 13:
                    grid[new_row][new_col] = snapshot[delta_row + 1][delta_col + 1]
    result_grid = [row[:] for row in grid]
    initial_snapshot = create_initial_snapshot(grid)
    for row in range(9):
        for col in range(4, 13):
            if grid[row][col] == 1:
                update_surrounding(result_grid, initial_snapshot, row, col)
    return result_grid
