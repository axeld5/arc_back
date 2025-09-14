def locate_target_area(grid, size, prohibited_values):
    for row in range(size):
        for col in range(size):
            if grid[row][col] not in prohibited_values:
                temp_row = row
                while temp_row < size and grid[temp_row][col] not in prohibited_values:
                    temp_row += 1
                height = temp_row - row
                temp_col = col
                while temp_col < size and grid[row][temp_col] not in prohibited_values:
                    temp_col += 1
                width = temp_col - col
                return (row, col, height, width)
    return (None, None, 0, 0)

def replace_target_area_in_grid(grid, size, target_start, target_dimensions):
    target_row, target_col = target_start
    target_height, target_width = target_dimensions
    for row in range(size - target_height + 1):
        for col in range(size - target_width + 1):
            if grid[row][col] == 5:
                for h in range(target_height):
                    for w in range(target_width):
                        grid[row + h][col + w] = grid[target_row + h][target_col + w]
    return grid

def p(grid, range_func=range):
    grid_size = len(grid)
    prohibited_values = {0, 5}
    target_row, target_col, target_height, target_width = locate_target_area(grid, grid_size, prohibited_values)
    if target_row is not None:
        grid = replace_target_area_in_grid(grid, grid_size, (target_row, target_col), (target_height, target_width))
    return grid
