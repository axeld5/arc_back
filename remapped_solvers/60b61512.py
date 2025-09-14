from itertools import product as cartesian_product

def fill_center_of_plus_signs(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    for row, col in cartesian_product(range(num_rows - 2), range(num_cols - 2)):
        if is_plus_sign(matrix, row, col):
            fill_center_with_seven(matrix, row, col)
    return matrix

def is_plus_sign(matrix, row, col):
    return all((4 in segment for segment in [matrix[row][col:col + 3], matrix[row + 2][col:col + 3], [matrix[r][col] for r in range(row, row + 3)], [matrix[r][col + 2] for r in range(row, row + 3)]]))

def fill_center_with_seven(matrix, row, col):
    for r, c in cartesian_product(range(row, row + 3), range(col, col + 3)):
        if matrix[r][c] == 0:
            matrix[r][c] += 7

def p(matrix, A=range):
    return fill_center_of_plus_signs(matrix)
