def flatten_and_filter_empty_rows(grid):
    flat_list = sum(grid, [])
    non_empty_values = [value for value in flat_list if value]
    return non_empty_values

def p(j):
    return [flatten_and_filter_empty_rows(j)]
