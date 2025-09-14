def update_right_edge(grid, last_row_index):
    for col in range(1, len(grid[0])):
        grid[last_row_index][col] = 4

def update_lower_diagonal(grid):
    for index in range(1, len(grid[0])):
        grid[len(grid) - index - 1][index] = 2

def p(grid):
    last_row_index = len(grid) - 1
    update_right_edge(grid, last_row_index)
    update_lower_diagonal(grid)
    return grid
