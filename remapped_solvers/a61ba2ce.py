def get_top_left_square_value(grid, row, col):
    value = grid[row][col]
    if value and grid[row + 1][col] == value and (grid[row][col + 1] == value):
        return value
    return 0

def get_bottom_left_square_value(grid, row, col):
    value = grid[row][col]
    if value and grid[row + 1][col] == value and (grid[row + 1][col + 1] == value):
        return value
    return 0

def get_top_right_square_value(grid, row, col):
    value = grid[row][col]
    if value and grid[row][col + 1] == value and (grid[row + 1][col + 1] == value):
        return value
    return 0

def get_bottom_right_square_value(grid, row, col):
    value = grid[row + 1][col + 1]
    if value and grid[row + 1][col] == value and (grid[row][col + 1] == value):
        return value
    return 0

def p(grid):
    grid_size = 12
    block_values = [0] * 16
    for row in range(grid_size):
        for col in range(grid_size):
            top_left_index = 0
            bottom_left_index = 8
            top_right_index = 2
            bottom_right_index = 11
            if (value := get_top_left_square_value(grid, row, col)):
                block_values[top_left_index] = block_values[1] = block_values[4] = value
            if (value := get_bottom_left_square_value(grid, row, col)):
                block_values[bottom_left_index] = block_values[12] = block_values[13] = value
            if (value := get_top_right_square_value(grid, row, col)):
                block_values[top_right_index] = block_values[3] = block_values[7] = value
            if (value := get_bottom_right_square_value(grid, row, col)):
                block_values[bottom_right_index] = block_values[14] = block_values[15] = value
    return [block_values[i:i + 4] for i in (0, 4, 8, 12)]
