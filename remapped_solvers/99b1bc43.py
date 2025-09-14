def increment_grid_values(grid, range_limit=range(4)):
    for row_index in range_limit:
        for col_index in range_limit:
            grid[row_index][col_index] += grid[row_index + 5][col_index]
            if grid[row_index][col_index] == 3:
                grid[row_index][col_index] = 0
            elif grid[row_index][col_index] > 0:
                grid[row_index][col_index] = 3
    return grid[:4]

def p(grid, range_limit=range(4)):
    return increment_grid_values(grid, range_limit)
