def flood_fill(start_row, start_col, grid, visited):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stack = [(start_row, start_col)]
    component = []
    visited[start_row][start_col] = True
    while stack:
        row, col = stack.pop()
        component.append((row, col))
        for dr, dc in directions:
            new_row, new_col = (row + dr, col + dc)
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                if grid[new_row][new_col] != 0 and (not visited[new_row][new_col]):
                    visited[new_row][new_col] = True
                    stack.append((new_row, new_col))
    return component

def find_valid_component(components, grid):
    for component in components:
        if any((grid[row][col] != 8 for row, col in component)):
            return component
    return None

def extract_pattern(valid_component, grid):
    min_row = min((row for row, _ in valid_component))
    min_col = min((col for _, col in valid_component))
    return {(row - min_row, col - min_col): grid[row][col] for row, col in valid_component}

def apply_pattern(valid_component, components, grid):
    pattern = extract_pattern(valid_component, grid)
    result_grid = [[0] * len(grid[0]) for _ in range(len(grid))]
    for component in components:
        if all((grid[row][col] == 8 for row, col in component)):
            min_row = min((row for row, _ in component))
            min_col = min((col for _, col in component))
            for row, col in component:
                offset = (row - min_row, col - min_col)
                result_grid[row][col] = pattern.get(offset, 0)
    return result_grid

def p(grid, range_func=range):
    height = width = 10
    visited = [[False] * width for _ in range_func(height)]
    components = []
    for row in range_func(height):
        for col in range_func(width):
            if grid[row][col] != 0 and (not visited[row][col]):
                component = flood_fill(row, col, grid, visited)
                components.append(component)
    valid_component = find_valid_component(components, grid)
    return apply_pattern(valid_component, components, grid)
