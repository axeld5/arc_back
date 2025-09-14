def p(grid):

    def is_non_zero_and_not_four(value):
        return value and value != 4

    def update_below_row(grid, current_row, col_index, value):
        grid[current_row + 1][col_index] = value

    def fill_columns_with_four(grid, current_row, col_index):
        for row_index in range(current_row + 1):
            grid[row_index][col_index % 2::2] = [4] * len(grid[row_index][col_index % 2::2])

    def process_grid(grid):
        for row_index, row in enumerate(grid):
            for col_index, value in enumerate(row):
                if is_non_zero_and_not_four(value):
                    update_below_row(grid, row_index, col_index, value)
                    fill_columns_with_four(grid, row_index, col_index)
                    return grid
    return process_grid(grid)
