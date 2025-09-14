from collections import Counter

def find_most_common_positive_value(flat_list):
    count = Counter(flat_list).most_common()
    most_common_positive = [item for item in count if item[0] > 0][-1][0]
    return most_common_positive

def filter_rows_with_value(grid, value):
    return [row for row in grid if value in row]

def get_value_indices_in_rows(grid, value):
    indices = []
    for row in grid:
        for i in range(len(row)):
            if row[i] == value:
                indices.append(i)
    return indices

def extract_sub_rows(grid, min_index, max_index):
    return [row[min_index:max_index + 1] for row in grid]

def p(grid):
    flattened_grid = [element for row in grid for element in row]
    most_common_positive = find_most_common_positive_value(flattened_grid)
    filtered_rows = filter_rows_with_value(grid, most_common_positive)
    common_value_indices = get_value_indices_in_rows(filtered_rows, most_common_positive)
    min_index = min(common_value_indices)
    max_index = max(common_value_indices)
    return extract_sub_rows(filtered_rows, min_index, max_index)
