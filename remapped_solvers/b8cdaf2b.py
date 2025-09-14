from collections import Counter

def find_least_common_element(grid):
    flattened_grid = [value for row in grid for value in row]
    element_count = Counter(flattened_grid)
    least_common_element = min(element_count, key=element_count.get)
    return least_common_element

def find_initial_coordinates(grid, target_value):
    initial_coords = {(i - 1, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == target_value}
    return initial_coords

def shoot_diagonal_coordinates(height, start_row, start_col, row_step, col_step):
    coordinates = set()
    row, col = (start_row, start_col)
    while 0 <= row < height and 0 <= col < height:
        coordinates.add((row, col))
        row += row_step
        col += col_step
    return coordinates

def p(grid):
    grid_height = len(grid)
    least_common_value = find_least_common_element(grid)
    initial_indices = find_initial_coordinates(grid, least_common_value)
    start_row = min((i for i, j in initial_indices))
    start_col = min((j for i, j in initial_indices))
    alternate_row = min((i for i, j in initial_indices))
    max_col = max((j for i, j in initial_indices))
    diagonals = shoot_diagonal_coordinates(grid_height, start_row, start_col, -1, -1) | shoot_diagonal_coordinates(grid_height, alternate_row, max_col, -1, 1)
    for i, j in diagonals:
        if grid[i][j] == 0:
            grid[i][j] = least_common_value
    return grid
