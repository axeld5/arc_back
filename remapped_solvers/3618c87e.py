def update_column(grid, col_index, size_range):
    for row_index in size_range:
        if grid[row_index][col_index] == 1:
            grid[row_index][col_index] = 0
            grid[4][col_index] = 1
    return grid

def p(j, size_range=range(5)):
    updated_grid = j[:]
    for col_index in size_range:
        updated_grid = update_column(updated_grid, col_index, size_range)
    return updated_grid
