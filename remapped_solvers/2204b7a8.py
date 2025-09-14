def p(grid):

    def is_vertical_symmetry(grid):
        return grid[0][0] == grid[0][9]

    def find_unique_middle_value(grid, exclusion_values):
        for row in grid:
            for value in row:
                if value != 0 and value not in exclusion_values:
                    return value
    grid_size = 10
    modified_grid = [row[:] for row in grid]
    if is_vertical_symmetry(grid):
        main_value, alternative_value = (grid[0][0], grid[9][0])
    else:
        main_value, alternative_value = (grid[0][0], grid[0][9])
    unique_middle_value = find_unique_middle_value(grid, {main_value, alternative_value})
    for row_index in range(grid_size):
        for col_index in range(grid_size):
            if grid[row_index][col_index] == unique_middle_value:
                diagonal_decision = (row_index, grid_size - 1 - row_index) if is_vertical_symmetry(grid) else (col_index, grid_size - 1 - col_index)
                if diagonal_decision[0] < diagonal_decision[1]:
                    modified_grid[row_index][col_index] = main_value
                else:
                    modified_grid[row_index][col_index] = alternative_value
    return modified_grid
