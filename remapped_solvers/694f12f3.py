def find_connected_components(grid):
    GRID_SIZE = 10
    visited = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    components = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if visited[i][j]:
                continue
            number = grid[i][j]
            stack = [(i, j)]
            visited[i][j] = 1
            component_coordinates = []
            while stack:
                row, col = stack.pop()
                component_coordinates.append((row, col))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row, next_col = (row + dx, col + dy)
                    if 0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE and (not visited[next_row][next_col]) and (grid[next_row][next_col] == number):
                        visited[next_row][next_col] = 1
                        stack.append((next_row, next_col))
            components.append((number, component_coordinates))
    return components

def get_bounds(coordinates):
    rows, cols = zip(*coordinates)
    return (min(rows), min(cols), max(rows), max(cols))

def shrink_bounds(bounds, padding=1):
    min_row, min_col, max_row, max_col = bounds
    return (min_row + padding, min_col + padding, max_row - padding, max_col - padding)

def get_coordinates_in_bounds(bounds):
    min_row, min_col, max_row, max_col = bounds
    return [(i, j) for i in range(min_row, max_row + 1) for j in range(min_col, max_col + 1)]

def p(grid, range_=range, min_=min, max_=max):
    components = find_connected_components(grid)
    fours_components = [coordinates for number, coordinates in components if number == 4]
    smallest_component = min_(fours_components, key=len)
    largest_component = max_(fours_components, key=len)
    modified_grid = [row[:] for row in grid]
    NEW_NUMBERS = {1: smallest_component, 2: largest_component}
    for new_num, component in NEW_NUMBERS.items():
        component_bounds = get_bounds(component)
        inner_bounds = shrink_bounds(component_bounds)
        for i, j in get_coordinates_in_bounds(inner_bounds):
            if 0 <= i < 10 and 0 <= j < 10:
                modified_grid[i][j] = new_num
    return modified_grid
