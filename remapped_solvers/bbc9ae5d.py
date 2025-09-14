def solve_half_iterations(grid):
    row_repeats = len(grid[0]) // 2
    expanded_grid = [row[:] for row in grid for _ in range(row_repeats)]
    return expanded_grid

def find_value_coordinates(expanded_grid, target_value):
    coordinates = []
    for row_index, row in enumerate(expanded_grid):
        for col_index, value in enumerate(row):
            if value == target_value:
                coordinates.append((row_index, col_index))
    return coordinates

def extend_diagonals(expanded_grid, start_coordinates, fill_value):
    for start_x, start_y in start_coordinates:
        for offset in range(43):
            i, j = (start_x + offset, start_y + offset)
            if 0 <= i < len(expanded_grid) and 0 <= j < len(expanded_grid[0]):
                expanded_grid[i][j] = fill_value

def p(grid):
    expanded_grid = solve_half_iterations(grid)
    target_value = next((value for row in grid for value in row if value))
    start_coordinates = find_value_coordinates(expanded_grid, target_value)
    extend_diagonals(expanded_grid, start_coordinates, target_value)
    return expanded_grid
