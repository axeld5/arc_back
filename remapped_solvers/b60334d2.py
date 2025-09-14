def solve_grid_problem(grid):
    RANGE_OF_INTEREST = range(1, 8)

    def initialize_empty_grid():
        return [[0] * 9 for _ in range(9)]

    def update_grid(new_grid, i, j, value):
        new_grid[i][j] = value

    def apply_rules_to_cell(new_grid, grid, i, j):
        if grid[i][j]:
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row or delta_col:
                        value_to_set = 1 if delta_row * delta_col == 0 else 5
                        update_grid(new_grid, i + delta_row, j + delta_col, value_to_set)
    new_grid = initialize_empty_grid()
    for row in RANGE_OF_INTEREST:
        for col in RANGE_OF_INTEREST:
            apply_rules_to_cell(new_grid, grid, row, col)
    return new_grid
p = solve_grid_problem
