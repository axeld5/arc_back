def process_cell(grid, row, col, num_rows, num_cols):
    current_value = grid[row][col]
    if current_value == 0:
        return
    has_one_horizontal_neighbor = sum((grid[row][col + offset] == current_value for offset in (1, -1))) == 1
    has_one_vertical_neighbor = sum((grid[row + offset][col] == current_value for offset in (1, -1))) == 1
    if not (has_one_horizontal_neighbor and has_one_vertical_neighbor):
        return
    vertical_direction = 2 * (grid[row + 1][col] == current_value) - 1
    horizontal_direction = 2 * (grid[row][col + 1] == current_value) - 1
    new_col, new_row = (col, row)
    if grid[row + vertical_direction][col + horizontal_direction] == current_value:
        return
    while 1 <= new_row < num_rows - 1 and 1 <= new_col < num_cols - 1:
        new_row -= vertical_direction
        new_col -= horizontal_direction
        grid[new_row][new_col] = grid[row + 2 * vertical_direction][col + 2 * horizontal_direction]

def p(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))
    for row in range(1, num_rows - 1):
        for col in range(1, num_cols - 1):
            process_cell(grid, row, col, num_rows, num_cols)
    return grid
