from collections import Counter

def p(matrix, enumerate_function=enumerate):

    def find_filled_positions(matrix):
        non_zero_positions = [(row, col) for row, row_values in enumerate_function(matrix) for col, value in enumerate_function(row_values) if value != 0]
        return non_zero_positions

    def most_frequent_value(positions):
        values_counter = Counter((matrix[row][col] for row, col in positions))
        most_common_value = values_counter.most_common(1)[0][0]
        return most_common_value

    def extract_subgrid_with_value(positions, target_value):
        target_positions = [(row, col) for row, col in positions if matrix[row][col] == target_value]
        min_row = min((row for row, _ in target_positions))
        max_row = max((row for row, _ in target_positions)) + 1
        min_col = min((col for _, col in target_positions))
        max_col = max((col for _, col in target_positions)) + 1
        subgrid = [matrix[row][min_col:max_col] for row in range(min_row, max_row)]
        return subgrid
    filled_positions = find_filled_positions(matrix)
    if not filled_positions:
        return []
    most_frequent = most_frequent_value(filled_positions)
    result_subgrid = extract_subgrid_with_value(filled_positions, most_frequent)
    return result_subgrid
