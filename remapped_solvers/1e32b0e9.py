def apply_transformation(value, reference_matrix_value, identity_value):
    if reference_matrix_value == value:
        return value
    elif reference_matrix_value:
        return identity_value
    else:
        return 0

def transform_grid(grid):
    subgrid_indices = range(3)
    transformed_grid = [row[:] for row in grid]
    identity_value = grid[5][0]
    reference_matrix = [[grid[row + 1][col + 1] for col in subgrid_indices] for row in subgrid_indices]
    starting_points = [(0, 6), (0, 12), (6, 0), (6, 6), (6, 12), (12, 0), (12, 6), (12, 12)]
    for start_row, start_col in starting_points:
        for sub_row in subgrid_indices:
            for sub_col in subgrid_indices:
                grid_row = start_row + sub_row + 1
                grid_col = start_col + sub_col + 1
                transformed_grid[grid_row][grid_col] = apply_transformation(grid[grid_row][grid_col], reference_matrix[sub_row][sub_col], identity_value)
    return transformed_grid

def p(grid):
    return transform_grid(grid)
