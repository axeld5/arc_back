def find_value_position(grid, value):
    flat_list = sum(grid, [])
    index = flat_list.index(value)
    return divmod(index, len(grid[0]))

def update_adjacent_positions(grid, row, col):
    if row > 0 and col > 0:
        grid[row - 1][col - 1] = 3
    if row < len(grid) - 1 and col > 0:
        grid[row + 1][col - 1] = 8
    if col < len(grid[0]) - 1 and row > 0:
        grid[row - 1][col + 1] = 6
    if row < len(grid) - 1 and col < len(grid[0]) - 1:
        grid[row + 1][col + 1] = 7

def p(grid):
    row, col = find_value_position(grid, 2)
    grid[row][col] = 0
    update_adjacent_positions(grid, row, col)
    return grid
