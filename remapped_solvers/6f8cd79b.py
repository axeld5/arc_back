def solve(input_grid):
    num_rows = len(input_grid)
    num_cols = len(input_grid[0])
    result_grid = create_grid_with_value(num_rows, num_cols, 0)
    fill_borders_with_value(result_grid, 8)
    return result_grid

def create_grid_with_value(rows, cols, value):
    return [[value for _ in range(cols)] for _ in range(rows)]

def fill_borders_with_value(grid, value):
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if is_border_cell(row, col, num_rows, num_cols):
                grid[row][col] = value

def is_border_cell(row, col, num_rows, num_cols):
    return row == 0 or row == num_rows - 1 or col == 0 or (col == num_cols - 1)
p = solve
