def transpose(matrix):
    rows, cols = (len(matrix), len(matrix[0]))
    return [[matrix[row][col] for row in range(rows)] for col in range(cols)]

def process_grid(matrix):
    num_rows, num_cols = (len(matrix), len(matrix[0]))
    if num_cols > num_rows:
        return transpose(process_grid(transpose(matrix)))
    index_of_2, first_index_of_3, last_index_of_3 = (0, num_rows, 0)
    for row_idx, row in enumerate(matrix):
        if row[0] == 2:
            index_of_2 = row_idx
        if any((value == 3 for value in row)):
            first_index_of_3 = min(first_index_of_3, row_idx)
            last_index_of_3 = row_idx
    if first_index_of_3 < index_of_2:
        return process_grid(matrix[::-1])[::-1]
    part1 = matrix[:index_of_2 + 1]
    part2 = matrix[first_index_of_3:last_index_of_3 + 1]
    separator = [[8] * num_cols]
    blank_rows_count = num_rows - index_of_2 + first_index_of_3 - last_index_of_3 - 3
    blank_rows = [[0] * num_cols] * blank_rows_count
    return part1 + part2 + separator + blank_rows
p = process_grid
