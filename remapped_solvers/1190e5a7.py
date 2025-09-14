def solve(grid, u=range, length=len):

    def get_unique_row_indexes(grid, n):
        return [i for i in u(n) if length(set(grid[i])) == 1]

    def get_unique_column_indexes(grid, n, m):
        return [j for j in u(m) if length(set((grid[i][j] for i in u(n)))) == 1]

    def find_non_unique_value(grid, n, m, unique_rows, unique_columns):
        return next((grid[i][j] for i in u(n) for j in u(m) if i not in unique_rows and j not in unique_columns))
    num_rows = length(grid)
    num_columns = length(grid[0])
    unique_rows = get_unique_row_indexes(grid, num_rows)
    unique_columns = get_unique_column_indexes(grid, num_rows, num_columns)
    non_unique_value = find_non_unique_value(grid, num_rows, num_columns, unique_rows, unique_columns)
    result_grid = [[non_unique_value] * (length(unique_columns) + 1) for _ in u(length(unique_rows) + 1)]
    return result_grid
p = solve
