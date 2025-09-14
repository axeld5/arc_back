def identify_columns_with_twos(grid):
    columns_with_twos = {column_index for row in grid for column_index, value in enumerate(row) if value == 2}
    return columns_with_twos

def update_value(value, column_index, row, columns_with_twos):
    if 1 in row:
        return 1
    if 3 in row:
        return 3
    if value < 1 and column_index in columns_with_twos:
        return 2
    return value

def update_grid(grid):
    columns_with_twos = identify_columns_with_twos(grid)
    updated_grid = [[update_value(value, col_index, row, columns_with_twos) for col_index, value in enumerate(row)] for row in grid]
    return updated_grid

def p(grid, enumeration_func=enumerate):
    return update_grid(grid)
