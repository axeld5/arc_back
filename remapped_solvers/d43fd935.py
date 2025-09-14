def find_first_block(grid, height):
    for row in range(height - 1):
        for col in range(height - 1):
            value = grid[row][col]
            if value != 0 and grid[row + 1][col] == value and (grid[row][col + 1] == value) and (grid[row + 1][col + 1] == value):
                return (row, col, value)
    return (None, None, None)

def propagate_value(grid, height, block_row, block_col, block_value):
    for row in range(height):
        for col in range(height):
            value = grid[row][col]
            if value == 0 or value == block_value:
                continue
            if row == block_row or row == block_row + 1:
                if col < block_col:
                    for fill_col in range(col + 1, block_col):
                        grid[row][fill_col] = value
                elif col > block_col + 1:
                    for fill_col in range(col - 1, block_col + 1, -1):
                        grid[row][fill_col] = value
            if col == block_col or col == block_col + 1:
                if row < block_row:
                    for fill_row in range(row + 1, block_row):
                        grid[fill_row][col] = value
                elif row > block_row + 1:
                    for fill_row in range(row - 1, block_row + 1, -1):
                        grid[fill_row][col] = value

def p(grid, height=10, range_function=range):
    block_row, block_col, block_value = find_first_block(grid, height)
    if block_value is not None:
        propagate_value(grid, height, block_row, block_col, block_value)
    return grid
