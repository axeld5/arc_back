def apply_cell_transformation(grid, row_index, col_index, transformation_map):
    num_rows = len(grid)
    num_cols = len(grid[0])
    for delta_row in range(-1, 2):
        for delta_col in range(-1, 2):
            if delta_row == 0 and delta_col == 0:
                continue
            new_row = row_index + delta_row
            new_col = col_index + delta_col
            if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
                grid[new_row][new_col] = transformation_map[grid[row_index][col_index]]

def transform_grid(grid):
    transformation_map = {8: 4, 2: 1, 3: 6}
    transformed_grid = [[cell for cell in row] for row in grid]
    for row_index, row in enumerate(grid):
        for col_index, cell_value in enumerate(row):
            if cell_value:
                apply_cell_transformation(transformed_grid, row_index, col_index, transformation_map)
    return transformed_grid

def p(j, A=enumerate):
    return transform_grid(j)
