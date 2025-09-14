def p(grid):
    height = len(grid)
    for row in range(height // 2):
        for col in range(len(grid[row])):
            grid[row][col] = grid[-(row + 1)][col]
    return grid
