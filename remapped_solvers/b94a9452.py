def p(grid):
    grid = [row for row in grid if sum(row) > 0]
    col_indices = []
    values = []
    for row in grid:
        for col_idx in range(len(row)):
            if row[col_idx] > 0:
                col_indices.append(col_idx)
                values.append(row[col_idx])
    unique_values = list(set(values))
    value_map = {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]}
    min_col, max_col = (min(col_indices), max(col_indices))
    result = []
    for row in grid:
        cropped_row = row[min_col:max_col + 1]
        mapped_row = [value_map[val] for val in cropped_row]
        result.append(mapped_row)
    return result
