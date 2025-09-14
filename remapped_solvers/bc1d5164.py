def update_maximums(grid, max_grid, row_offset, col_offset):
    for row in range(2):
        for col in range(2):
            effective_row = row + row_offset
            effective_col = col + col_offset
            max_grid[effective_row][effective_col] = max(max_grid[effective_row][effective_col], grid[effective_row][effective_col])

def get_maximum_subgrid(grid):
    max_grid = [[0] * 3 for _ in range(3)]
    for configuration in range(16):
        row_offset = configuration // 8 % 2 * -2 + configuration // 2 % 2
        col_offset = configuration // 4 % 2 * -2 + configuration % 2
        update_maximums(grid, max_grid, row_offset, col_offset)
    return max_grid

def p(j):
    return get_maximum_subgrid(j)
