def reverse_and_concatenate(row):
    return row[::-1] + row

def transform_grid_part(grid_part):
    return [reverse_and_concatenate(row) for row in grid_part]

def solve(grid):
    transformed_grid = transform_grid_part(grid)
    selected_rows = [transformed_grid[2], transformed_grid[1], transformed_grid[0]]
    return selected_rows + transformed_grid + selected_rows
p = solve
