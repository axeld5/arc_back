def solve(grid):
    size = len(grid)

    def find_first_nonzero_value(grid):
        for row in range(size):
            for col in range(size):
                if grid[row][col]:
                    return grid[row][col]

    def get_positions_of_value(grid, value):
        positions = [(row, col) for row in range(size) for col in range(size) if grid[row][col] == value]
        return sorted(positions)

    def create_expanded_grid(primary_value, positions):
        expanded_size = 3 * (size * size - len(positions))
        expanded_grid = [[0] * expanded_size for _ in range(expanded_size)]
        for index, (row, col) in enumerate(positions):
            base_row = index // (size * size - len(positions)) * size
            base_col = index % (size * size - len(positions)) * size
            for pr, pc in positions:
                if 0 <= base_row + pr < expanded_size and 0 <= base_col + pc < expanded_size:
                    expanded_grid[base_row + pr][base_col + pc] = primary_value
        return expanded_grid
    primary_value = find_first_nonzero_value(grid)
    positions = get_positions_of_value(grid, primary_value)
    solution = create_expanded_grid(primary_value, positions)
    return solution
p = solve
