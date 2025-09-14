def p(input_grid, u=range):

    def get_expanded_grid(original_grid, expansion_factor):
        rows = len(original_grid)
        columns = len(original_grid[0])
        return [[original_grid[i // expansion_factor][j // expansion_factor] for j in u(columns * expansion_factor)] for i in u(rows * expansion_factor)]

    def mark_border(grid, row_start, col_start, square_size, grid_rows, grid_cols):
        for dr, dc in [(-1, -1), (-1, square_size), (square_size, -1), (square_size, square_size)]:
            row, col = (row_start + dr, col_start + dc)
            while -1 < row < grid_rows and -1 < col < grid_cols and (not grid[row][col]):
                grid[row][col] = 2
                row += (dr > 0) - (dr < 0)
                col += (dc > 0) - (dc < 0)

    def find_and_mark_largest_square(expanded_grid, original_rows, original_cols, expansion_factor):
        expanded_rows = original_rows * expansion_factor
        expanded_cols = original_cols * expansion_factor
        for size in u(min(expanded_rows, expanded_cols), 0, -1):
            for row in u(expanded_rows - size + 1):
                for col in u(expanded_cols - size + 1):
                    initial_value = expanded_grid[row][col]
                    if initial_value and all((line[col:col + size] == [initial_value] * size for line in expanded_grid[row:row + size])):
                        mark_border(expanded_grid, row, col, size, expanded_rows, expanded_cols)
                        return expanded_grid
    unique_nonzero_count = len(set(sum(input_grid, [])) - {0})
    expanded_grid = get_expanded_grid(input_grid, unique_nonzero_count)
    return find_and_mark_largest_square(expanded_grid, len(input_grid), len(input_grid[0]), unique_nonzero_count)
