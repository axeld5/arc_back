def repeat_grid_rows(grid: list) -> list:
    num_original_rows = len(grid)
    num_original_columns = len(grid[0])
    repeated_grid = []
    for i in range(num_original_rows * 1):
        row_index = i % 5
        new_row = []
        for j in range(num_original_columns * 2):
            column_index = j % 6
            new_row.append(grid[row_index][column_index])
        repeated_grid.append(new_row)
    return repeated_grid

def p(grid: list) -> list:
    return repeat_grid_rows(grid)
