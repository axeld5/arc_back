def find_indices_in_grid(grid, target):
    return [index for index, row in enumerate(grid) if target in row]

def calculate_number_of_zeros_and_eights(row_5_indices, row_8_indices):
    difference_in_lengths = len(row_5_indices) - 1 - len(row_8_indices)
    return [8] * difference_in_lengths + [0] * (6 - difference_in_lengths)

def p(grid):
    indices_with_5 = find_indices_in_grid(grid, 5)
    indices_with_8 = find_indices_in_grid(grid, 8)
    result_row = calculate_number_of_zeros_and_eights(indices_with_5, indices_with_8)
    return [result_row[:3], result_row[5:2:-1], [0, 0, 0]]
