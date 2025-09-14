def update_diagonals(grid, row, col, length, value):
    directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    for direction in directions:
        current_row, current_col = (row + direction[0], col + direction[1])
        while 0 <= current_row < length and 0 <= current_col < length:
            grid[current_row][current_col] = value
            current_row += direction[0]
            current_col += direction[1]

def p(initial_grid):
    grid = [row[:] for row in initial_grid]
    grid_size = len(initial_grid)
    for row in range(grid_size):
        for col in range(grid_size):
            if initial_grid[row][col]:
                value = initial_grid[row][col]
                update_diagonals(grid, row, col, grid_size, value)
    return grid
