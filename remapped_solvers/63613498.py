from collections import deque

def get_neighboring_positions(row, col, max_size):
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        new_row, new_col = (row + delta_row, col + delta_col)
        if 0 <= new_row < max_size and 0 <= new_col < max_size:
            yield (new_row, new_col)

def find_connected_components(grid, max_size):
    visited_positions = set()
    connected_components = []
    for row in range(max_size):
        for col in range(max_size):
            if (row, col) in visited_positions or grid[row][col] == 0:
                continue
            component_value = grid[row][col]
            queue = deque([(row, col)])
            component_positions = set()
            while queue:
                current_row, current_col = queue.popleft()
                if (current_row, current_col) in visited_positions or grid[current_row][current_col] != component_value:
                    continue
                visited_positions.add((current_row, current_col))
                component_positions.add((current_row, current_col))
                queue.extend(get_neighboring_positions(current_row, current_col, max_size))
            connected_components.append(component_positions)
    return connected_components

def extract_shape_and_shift(grid, positions):
    min_row = min((row for row, _ in positions))
    min_col = min((col for _, col in positions))
    normalized_shape = {(row - min_row, col - min_col) for row, col in positions}
    return normalized_shape

def overlay_pattern_on_grid(grid, positions, max_value, overlay_value):
    new_grid = [list(row) for row in grid]
    for row, col in positions:
        new_grid[row][col] = overlay_value
    for row in range(max_value):
        for col in range(max_value):
            new_grid[row][col] = grid[row][col]
    return new_grid

def p(grid, minimum=min):
    MAX_SIZE = 10
    active_positions = {(row, col) for row in range(minimum(3, MAX_SIZE)) for col in range(minimum(3, MAX_SIZE)) if grid[row][col]}
    min_row, min_col = (minimum((row for row, _ in active_positions)), minimum((col for _, col in active_positions)))
    target_shape = {(row - min_row, col - min_col) for row, col in active_positions}
    found_components = find_connected_components(grid, MAX_SIZE)
    matching_positions = set()
    for component in found_components:
        normalized_shape = extract_shape_and_shift(grid, component)
        if normalized_shape == target_shape:
            matching_positions |= component
    return overlay_pattern_on_grid(grid, matching_positions, minimum(3, MAX_SIZE), 5)
