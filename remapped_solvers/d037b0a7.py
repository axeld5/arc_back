def fill_column_with_first_non_zero(grid, range_function=range):
    for column in range_function(3):
        first_non_zero_value = None
        first_non_zero_row = None
        for row in range_function(3):
            if grid[row][column]:
                first_non_zero_value = grid[row][column]
                first_non_zero_row = row
                break
        if first_non_zero_value is not None:
            for row in range_function(first_non_zero_row, 3):
                grid[row][column] = first_non_zero_value
    return grid

def p(j, A=range):
    return fill_column_with_first_non_zero(j, A)
