def rotate_90_degrees_clockwise(grid):
    return [list(row) for row in zip(*grid[::-1])]

def sum_matrices(matrix1, matrix2):
    return [row1 + row2 for row1, row2 in zip(matrix1, matrix2)]

def p(grid):
    rotated_90 = rotate_90_degrees_clockwise(grid)
    rotated_180 = rotate_90_degrees_clockwise(rotated_90)
    rotated_270 = rotate_90_degrees_clockwise(rotated_180)
    sum_original_and_90 = sum_matrices(grid, rotated_90)
    sum_270_and_180 = sum_matrices(rotated_270, rotated_180)
    return sum_original_and_90 + sum_270_and_180
