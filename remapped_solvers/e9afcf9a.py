def solve(grid):

    def select_value(row_index, column_index, grid):
        return grid[row_index if column_index % 2 == 0 else 1 - row_index][column_index]

    def transform_row(row_index, grid):
        return [select_value(row_index, column_index, grid) for column_index in range(6)]
    return [transform_row(row_index, grid) for row_index in range(2)]
p = solve
