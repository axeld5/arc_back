from collections import Counter as Q

def p(grid, range_func=range):
    HEIGHT = 6
    WIDTH = range_func(HEIGHT)
    row_sums = [sum((1 for value in grid[row_index] if value)) for row_index in WIDTH]
    column_sums = [sum((1 for row_index in WIDTH if grid[row_index][column_index])) for column_index in WIDTH]
    max_row_index = max(WIDTH, key=lambda i: (row_sums[i], -i))
    max_column_index = max(WIDTH, key=lambda j: (column_sums[j], -j))
    most_common_row_value = get_most_common_value(grid[max_row_index]) if row_sums[max_row_index] else 0
    most_common_column_value = get_most_common_value([grid[i][max_column_index] for i in WIDTH]) if column_sums[max_column_index] else 0
    output_grid = [[0] * HEIGHT for _ in WIDTH]
    if column_sums[max_column_index]:
        for i in WIDTH:
            output_grid[i][max_column_index] = most_common_column_value
    if row_sums[max_row_index]:
        for j in WIDTH:
            output_grid[max_row_index][j] = most_common_row_value
    used_values = {value for row in grid for value in row if value}
    third_value = find_unique_not_in(most_common_row_value, most_common_column_value, used_values, range_func)
    if row_sums[max_row_index] and column_sums[max_column_index]:
        output_grid[max_row_index][max_column_index] = third_value
    return output_grid

def get_most_common_value(elements):
    return Q((value for value in elements if value)).most_common(1)[0][0]

def find_unique_not_in(value1, value2, used_values, range_func):
    return next((x for x in used_values if x not in (value1, value2)), None) or (4 if 4 not in (value1, value2) else next((x for x in range_func(1, 10) if x not in (value1, value2))))
