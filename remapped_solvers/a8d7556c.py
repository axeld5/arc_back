def p(grid, range_=range, size=18):

    def is_block_zero(row, col):
        return grid[row][col] + grid[row][col + 1] + grid[row + 1][col] + grid[row + 1][col + 1] == 0

    def update_block(blocks):
        for row, col in blocks:
            grid[row][col] = 2
            grid[row][col + 1] = 2
            grid[row + 1][col] = 2
            grid[row + 1][col + 1] = 2
    first_attempt_blocks = []
    for row in range_(size - 1):
        for col in range_(size - 1):
            if is_block_zero(row, col):
                if (col < 1 or grid[row][col - 1] + grid[row + 1][col - 1]) and (col > size - 3 or grid[row][col + 2] + grid[row + 1][col + 2]):
                    first_attempt_blocks.append((row, col))
    update_block(first_attempt_blocks)
    second_attempt_blocks = []
    for row in range_(size - 1):
        for col in range_(size - 1):
            if is_block_zero(row, col):
                if (row < 1 or grid[row - 1][col] + grid[row - 1][col + 1]) and (row > size - 3 or grid[row + 2][col] + grid[row + 2][col + 1]):
                    second_attempt_blocks.append((row, col))
    update_block(second_attempt_blocks)
    return grid
