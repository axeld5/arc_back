def get_max_in_subgrid(grid, start_row, start_col):
    return max(grid[start_row][start_col], grid[start_row][start_col + 1], grid[start_row + 1][start_col], grid[start_row + 1][start_col + 1])

def reduce_grid(grid):
    original_rows = len(grid)
    original_cols = len(grid[0])
    reduced_grid = [[get_max_in_subgrid(grid, row, col) for col in range(0, original_cols, 2)] for row in range(0, original_rows, 2)]
    return reduced_grid

def expand_grid(reduced_grid, original_rows, original_cols):
    expanded_grid = [[reduced_grid[row // 4][col // 4] for col in range(2 * original_cols)] for row in range(2 * original_rows)]
    return expanded_grid

def p(grid, range_generator=range):
    c = len(grid)
    E = len(grid[0])
    reduced_grid = reduce_grid(grid)
    expanded_grid = expand_grid(reduced_grid, c, E)
    return expanded_grid
