def find_symbol_for_cross(grid):
    size = len(grid)
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            center_symbol = grid[i][j]
            if center_symbol and grid[i - 1][j - 1] == center_symbol == grid[i - 1][j + 1] == grid[i + 1][j - 1] == grid[i + 1][j + 1]:
                if is_valid_cross(grid, i, j, center_symbol):
                    return (i, j, center_symbol)
    return (None, None, None)

def is_valid_cross(grid, i, j, symbol):
    size = len(grid)
    for delta_i in range(-2, 3):
        for delta_j in range(-2, 3):
            row = i + delta_i
            col = j + delta_j
            if 0 <= row < size and 0 <= col < size and (grid[row][col] == symbol):
                if not (delta_i == 0 and delta_j == 0 or (abs(delta_i) == 1 and abs(delta_j) == 1)):
                    return False
    return True

def find_other_symbol(grid, cross_symbol):
    return next((symbol for row in grid for symbol in row if symbol and symbol != cross_symbol), None)

def transform_grid(grid, i, j, cross_symbol, other_symbol):
    size = len(grid)
    transformed_grid = [row[:] for row in grid]
    for row in range(size):
        for col in range(size):
            if grid[row][col] == other_symbol:
                row_mirror, col_mirror = (2 * i - row, 2 * j - col)
                for new_row, new_col in ((row, col_mirror), (row_mirror, col), (row_mirror, col_mirror)):
                    if 0 <= new_row < size and 0 <= new_col < size:
                        transformed_grid[new_row][new_col] = other_symbol
    return transformed_grid

def p(grid):
    cross_row, cross_col, cross_symbol = find_symbol_for_cross(grid)
    if cross_symbol is None:
        return grid
    other_symbol = find_other_symbol(grid, cross_symbol)
    transformed_grid = transform_grid(grid, cross_row, cross_col, cross_symbol, other_symbol)
    return transformed_grid
