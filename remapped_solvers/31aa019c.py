def find_least_frequent_value(grid):
    from collections import Counter
    flat_list = [value for row in grid for value in row]
    value_counts = Counter(flat_list)
    least_frequent_value = min(value_counts, key=value_counts.get)
    return least_frequent_value

def find_first_occurrence(grid, target_value):
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == target_value:
                return (i, j)
    return (None, None)

def create_output_grid(row, col, target_value):
    output_grid = [[0] * 10 for _ in range(10)]
    if row is None or col is None:
        return output_grid
    output_grid[row][col] = target_value
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row != 0 or delta_col != 0:
                adjacent_row = row + delta_row
                adjacent_col = col + delta_col
                if 0 <= adjacent_row < 10 and 0 <= adjacent_col < 10:
                    output_grid[adjacent_row][adjacent_col] = 2
    return output_grid

def p(grid):
    least_frequent_value = find_least_frequent_value(grid)
    row, col = find_first_occurrence(grid, least_frequent_value)
    output_grid = create_output_grid(row, col, least_frequent_value)
    return output_grid
