def solve_grid(input_grid):

    def transform_value(top_value, bottom_value):
        return 3 if top_value + bottom_value == 2 else 0
    transformed_grid = []
    for row_index in range(6):
        transformed_row = []
        for col_index in range(5):
            transformed_value = transform_value(input_grid[row_index][col_index], input_grid[row_index + 7][col_index])
            transformed_row.append(transformed_value)
        transformed_grid.append(transformed_row)
    return transformed_grid
p = solve_grid
