def p(grid, enumerated_function=enumerate):

    def is_valid_pair(val1, val2):
        return val1 == val2 or val1 * val2 < 1

    def find_repeating_column_pattern():
        num_rows = len(grid)
        num_columns = len(grid[0])
        for col_offset in range(1, num_columns):
            if all((is_valid_pair(current, next_val) for row in grid for current, next_val in zip(row, row[col_offset:]))):
                return col_offset
        return num_columns

    def find_repeating_row_pattern():
        num_rows = len(grid)
        for row_offset in range(1, num_rows):
            if all((is_valid_pair(current, next_val) for current_row, next_row in zip(grid, grid[row_offset:]) for current, next_val in zip(current_row, next_row))):
                return row_offset
        return num_rows
    column_pattern_length = find_repeating_column_pattern()
    row_pattern_length = find_repeating_row_pattern()
    cache = {}
    for row_idx, row in enumerated_function(grid):
        for col_idx, value in enumerated_function(row):
            if value:
                cache[row_idx % row_pattern_length, col_idx % column_pattern_length] = value
    for row_idx, row in enumerated_function(grid):
        for col_idx, value in enumerated_function(row):
            if not value:
                row[col_idx] = cache[row_idx % row_pattern_length, col_idx % column_pattern_length]
    return grid
