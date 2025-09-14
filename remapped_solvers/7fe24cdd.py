def rotate_matrix_90_degrees_clockwise(matrix):
    return [[*row] for row in zip(*matrix[::-1])]

def get_column_wise_sums(original, rotated_90, rotated_180, rotated_270):
    sums_90 = [a + b for a, b in zip(original, rotated_90)]
    sums_180 = [a + b for a, b in zip(rotated_270, rotated_180)]
    return sums_90 + sums_180

def p(grid):
    rotated_90 = rotate_matrix_90_degrees_clockwise(grid)
    rotated_180 = rotate_matrix_90_degrees_clockwise(rotated_90)
    rotated_270 = rotate_matrix_90_degrees_clockwise(rotated_180)
    return get_column_wise_sums(grid, rotated_90, rotated_180, rotated_270)
