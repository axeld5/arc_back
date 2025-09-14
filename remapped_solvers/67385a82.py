def copy_grid(grid):
    return [row[:] for row in grid]

def is_valid_position(row, col, max_row, max_col):
    return 0 <= row < max_row and 0 <= col < max_col

def mark_adjacent_eights(grid, result_grid, row, col, max_row, max_col):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for d_row, d_col in directions:
        new_row, new_col = (row + d_row, col + d_col)
        if is_valid_position(new_row, new_col, max_row, max_col) and grid[new_row][new_col] == 3:
            result_grid[row][col] = 8
            break

def p(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))
    result_grid = copy_grid(grid)
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] == 3:
                mark_adjacent_eights(grid, result_grid, row, col, num_rows, num_cols)
    return result_grid
