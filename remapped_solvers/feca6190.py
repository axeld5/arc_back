def create_output_matrix(size):
    return [[0] * size for _ in range(size)]

def fill_diagonals(output_matrix, column_values, starting_col):
    matrix_size = len(output_matrix)
    for current_col, col_value in enumerate(column_values):
        for row in range(current_col, matrix_size):
            diagonal_index = matrix_size - 1 + current_col - row
            output_matrix[row][diagonal_index] = col_value

def p(grid, R=range):
    first_row = grid[0]
    non_zero_count = sum((1 for value in first_row if value != 0))
    matrix_size = non_zero_count * 5
    output_matrix = create_output_matrix(matrix_size)
    fill_diagonals(output_matrix, first_row, 0)
    return output_matrix
