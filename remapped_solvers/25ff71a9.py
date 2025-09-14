def rotate_left(row, positions):
    actual_positions = positions % len(row)
    return row[actual_positions:] + row[:actual_positions]

def transform_grid(grid):
    rearranged_grid = grid[2:] + grid[:2]
    transformed_grid = [rotate_left(row, 3) for row in rearranged_grid]
    return transformed_grid

def p(grid):
    return transform_grid(grid)
