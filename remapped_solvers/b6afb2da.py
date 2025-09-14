def fill_grid_area_with_value(grid, start_col, start_row, end_col, end_row, value):
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            grid[row][col] = value

def mark_boundary(grid, start_col, start_row, end_col, end_row):
    fill_grid_area_with_value(grid, start_col, start_row, end_col, end_row, 4)
    fill_grid_area_with_value(grid, start_col + 1, start_row + 1, end_col - 1, end_row - 1, 2)
    grid[start_row][start_col] = grid[start_row][end_col] = 1
    grid[end_row][start_col] = grid[end_row][end_col] = 1

def p(grid):
    GRID_SIZE = 10
    for index in range(GRID_SIZE * GRID_SIZE):
        col, row = (index % GRID_SIZE, index // GRID_SIZE)
        if grid[row][col] == 5:
            end_col, end_row = (col, row)
            while end_col < GRID_SIZE - 1 and grid[end_row][end_col + 1] == 5:
                end_col += 1
            while end_row < GRID_SIZE - 1 and grid[end_row + 1][end_col] == 5:
                end_row += 1
            mark_boundary(grid, col, row, end_col, end_row)
    return grid
