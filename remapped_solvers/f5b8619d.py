def solve(grid):

    def find_non_zero_columns(grid, size):
        return {col for row in range(size) for col in range(size) if grid[row][col]}

    def process_grid(grid, size, non_zero_columns):
        return [[8 if grid[row][col] == 0 and col in non_zero_columns else grid[row][col] for col in range(size)] for row in range(size)]

    def expand_grid(modified_grid, size):
        new_size = 2 * size
        return [[modified_grid[row % size][col % size] for col in range(new_size)] for row in range(new_size)]
    size = len(grid)
    non_zero_columns = find_non_zero_columns(grid, size)
    modified_grid = process_grid(grid, size, non_zero_columns)
    expanded_grid = expand_grid(modified_grid, size)
    return expanded_grid
p = solve
