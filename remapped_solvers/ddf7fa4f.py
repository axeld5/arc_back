def p(grid):
    filled_grid = [row[:] for row in grid]

    def fill_region(start_row, start_col, new_value):
        if 0 <= start_row < 10 and 0 <= start_col < 10 and (filled_grid[start_row][start_col] == 5):
            filled_grid[start_row][start_col] = new_value
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                fill_region(start_row + delta_row, start_col + delta_col, new_value)
    for col in range(10):
        fill_value = grid[0][col]
        if fill_value:
            for row in range(1, 10):
                if filled_grid[row][col] == 5:
                    fill_region(row, col, fill_value)
    return filled_grid
