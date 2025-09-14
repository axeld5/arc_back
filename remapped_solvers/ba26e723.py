def replace_number_in_column(matrix, column_index, target_number, replacement_number):
    for row in matrix:
        if row[column_index] == target_number:
            row[column_index] = replacement_number

def solve(matrix):
    for column_index in range(0, len(matrix[0]), 3):
        replace_number_in_column(matrix, column_index, 4, 6)
    return matrix
p = solve
