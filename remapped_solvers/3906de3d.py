def p(grid, range_func=range):
    rows_count = len(grid)
    columns_count = len(grid[0])
    transformed_grid = [[0] * columns_count for _ in range_func(rows_count)]
    for col in range_func(columns_count):
        non_zero_elements = get_non_zero_elements_in_column(grid, col, range_func)
        fill_column_with_elements(transformed_grid, col, non_zero_elements)
    return transformed_grid

def get_non_zero_elements_in_column(grid, column_index, range_func):
    return [grid[row_index][column_index] for row_index in range_func(len(grid)) if grid[row_index][column_index] != 0]

def fill_column_with_elements(grid, column_index, elements):
    for element_index, element in enumerate(elements):
        grid[element_index][column_index] = element
