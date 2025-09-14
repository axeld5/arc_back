def p(grid, range_func=range):
    grid_size = len(grid)
    value_positions = {}
    for row in range_func(grid_size):
        for column in range_func(grid_size):
            value = grid[row][column]
            if value:
                if value not in value_positions:
                    value_positions[value] = []
                value_positions[value].append((row, column))
    most_frequent_value = max(value_positions, key=lambda x: len(value_positions[x]))
    positions_of_most_frequent = value_positions[most_frequent_value]
    u, w = find_colinear_points(positions_of_most_frequent)
    step = calculate_step_size(positions_of_most_frequent, u, w)
    transformed_grid = [row[:] for row in grid]
    transformed_grid[u][w] = most_frequent_value
    spread_value(transformed_grid, most_frequent_value, u, w, step, grid_size, range_func)
    return transformed_grid

def find_colinear_points(positions):
    for a, b in ((positions[0], positions[1]), (positions[1], positions[2]), (positions[2], positions[0])):
        mid_x = (a[0] + b[0]) // 2
        mid_y = (a[1] + b[1]) // 2
        if (a[0] + b[0]) % 2 == 0 and (a[1] + b[1]) % 2 == 0 and ((mid_x, mid_y) in positions):
            return (mid_x, mid_y)
    return (None, None)

def calculate_step_size(positions, x, y):
    for a, b in ((positions[0], positions[1]), (positions[1], positions[2]), (positions[2], positions[0])):
        if (a[0] + b[0]) // 2 == x and (a[1] + b[1]) // 2 == y:
            return max(abs(a[0] - b[0]), abs(a[1] - b[1])) // 2
    return 0

def spread_value(grid, value, center_x, center_y, step, size, range_func):
    current_step = step
    while current_step <= size + size:
        for offset in range_func(-current_step, current_step + 1):
            for row, column in ((center_x - current_step, center_y + offset), (center_x + current_step, center_y + offset), (center_x + offset, center_y - current_step), (center_x + offset, center_y + current_step)):
                if 0 <= row < size and 0 <= column < size:
                    grid[row][column] = value
        current_step += step
