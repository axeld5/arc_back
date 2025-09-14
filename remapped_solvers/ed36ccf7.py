def rotate_grid_90_degrees_clockwise(grid):
    transposed_grid = transpose_grid(grid)
    rotated_grid = reverse_rows(transposed_grid)
    return rotated_grid

def transpose_grid(grid):
    return [list(row) for row in zip(*grid)]

def reverse_rows(grid):
    return grid[::-1]

def p(grid):
    return rotate_grid_90_degrees_clockwise(grid)
