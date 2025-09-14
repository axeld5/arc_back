def rotate_clockwise(grid):
    num_rows = len(grid)
    rotated_grid = [[3] * num_rows, [0] * (num_rows - 1) + [3]]
    rotated_grid += [list(row) for row in zip(*grid[::-1])]
    return rotated_grid

def p(input_grid):
    width = len(input_grid[0])
    if width % 2 == 0:
        base_pattern = [[3, 3, 3, 3], [0, 0, 0, 3], [3, 3, 0, 3], [3, 3, 3, 3]]
    else:
        base_pattern = [[3, 3, 3], [0, 0, 3], [3, 0, 3], [3, 0, 3], [3, 3, 3]]
    num_rotations = max(0, width - 4)
    for _ in range(num_rotations):
        base_pattern = rotate_clockwise(base_pattern)
    return base_pattern
