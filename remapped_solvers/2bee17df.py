def fill_column_with_threes(grid, col_index, size):
    for row in range(1, size - 1):
        grid[row][col_index] = 3

def fill_row_with_threes(grid, row_index, size):
    for col in range(1, size - 1):
        if grid[row_index][col] == 0:
            grid[row_index][col] = 3

def solve(grid):
    size = len(grid)
    result_grid = [row[:] for row in grid]
    for col in range(size):
        if grid[1][col] == 0 and grid[size - 2][col] == 0 and (sum((grid[row][col] for row in range(1, size - 1))) == 0):
            fill_column_with_threes(result_grid, col, size)
    for row in range(size):
        if grid[row][1] == 0 and grid[row][size - 2] == 0 and (sum((grid[row][col] for col in range(1, size - 1))) == 0):
            fill_row_with_threes(result_grid, row, size)
    return result_grid
p = solve
