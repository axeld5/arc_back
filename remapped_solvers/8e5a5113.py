def rotate_grid_clockwise(matrix):
    return [list(row) for row in zip(*matrix[::-1])]

def rotate_grid_counterclockwise(matrix):
    return [row[::-1] for row in matrix[::-1]]

def apply_rotations(grids):
    top_left_3x3 = [row[:3] for row in grids[:3]]
    rotated_clockwise = rotate_grid_clockwise(top_left_3x3)
    rotated_counterclockwise = rotate_grid_counterclockwise(top_left_3x3)
    for i in range(3):
        for j in range(3):
            grids[i][j + 4] = rotated_clockwise[i][j]
            grids[i][j + 8] = rotated_counterclockwise[i][j]
    return grids

def p(g):
    return apply_rotations(g)
