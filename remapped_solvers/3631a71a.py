def p(grid, range_constructor=range):
    BOUNDS_CHECK_VALUE = 32
    TARGET_VALUE = 9
    grid_height = len(grid)
    grid_width = len(grid[0])
    modified_grid = grid[:]

    def is_within_bounds(row, col):
        return 0 <= row < grid_height and 0 <= col < grid_width

    def get_symmetric_positions(row, col):
        possible_positions = [(row, col), (col, row), (row, BOUNDS_CHECK_VALUE - 1 - col), (col, BOUNDS_CHECK_VALUE - 1 - row), (BOUNDS_CHECK_VALUE - 1 - row, col), (BOUNDS_CHECK_VALUE - 1 - col, row), (BOUNDS_CHECK_VALUE - 1 - row, BOUNDS_CHECK_VALUE - 1 - col), (BOUNDS_CHECK_VALUE - 1 - col, BOUNDS_CHECK_VALUE - 1 - row)]
        return possible_positions
    for row in range_constructor(grid_height):
        for col in range_constructor(grid_width):
            if grid[row][col] == TARGET_VALUE:
                for sym_row, sym_col in get_symmetric_positions(row, col):
                    if is_within_bounds(sym_row, sym_col) and grid[sym_row][sym_col] != TARGET_VALUE:
                        modified_grid[row][col] = grid[sym_row][sym_col]
                        break
    return modified_grid
