def merge_grids(grid):
    grid_size = 4
    merged_grid = []
    for x in range(grid_size):
        new_row = []
        for y in range(grid_size):
            merged_value = grid[x][y + grid_size] or grid[x + grid_size][y] or grid[x + grid_size][y + grid_size] or grid[x][y]
            new_row.append(merged_value)
        merged_grid.append(new_row)
    return merged_grid

def p(grid, range_=range(4)):
    return merge_grids(grid)
