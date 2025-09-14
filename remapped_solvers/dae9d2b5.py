def apply_transformation(matrix, range_sequence=range(3)):
    for row_index in range_sequence:
        for col_index in range_sequence:
            matrix[row_index][col_index] += matrix[row_index][col_index + 3]
            if matrix[row_index][col_index] > 0:
                matrix[row_index][col_index] = 6

def extract_submatrix(matrix):
    return [row[:3] for row in matrix]

def p(grid, A=range(3)):
    apply_transformation(grid, A)
    return extract_submatrix(grid)
