def p(grid, Z=zip):

    def get_transposed(grid):
        return list(Z(*grid))

    def max_combine(original, transposed):
        return [[max(a, b) for a, b in Z(row_orig, row_trans)] for row_orig, row_trans in Z(original, transposed)]

    def find_most_frequent_non_zero(flat_grid):
        flattened = [value for row in flat_grid for value in row]
        return max(set(flattened), key=lambda x: (flattened.count(x), x) if x != 0 else (0, 0))

    def replace_zeros_with_most_frequent(grid, most_frequent):
        return [[most_frequent if value == 0 else value for value in row] for row in grid]

    def set_diagonal(grid, value):
        for k in range(min(len(grid), len(grid[0]))):
            grid[k][k] = value
    transposed_grid = get_transposed(grid)
    combined_grid = max_combine(grid, transposed_grid)
    most_frequent_value = find_most_frequent_non_zero(combined_grid)
    updated_grid = replace_zeros_with_most_frequent(combined_grid, most_frequent_value)
    set_diagonal(updated_grid, updated_grid[0][0])
    return updated_grid
