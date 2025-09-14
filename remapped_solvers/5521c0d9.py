def find_connected_component(grid, start_row, start_col, label):
    stack = [(start_row, start_col)]
    component = [(start_row, start_col)]
    visited = set(component)
    while stack:
        row, col = stack.pop()
        for d_row, d_col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = (row + d_row, col + d_col)
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]) and (grid[new_row][new_col] == label) and ((new_row, new_col) not in visited):
                visited.add((new_row, new_col))
                stack.append((new_row, new_col))
                component.append((new_row, new_col))
    return component

def process_component(component, cleaned_grid, grid):
    min_row = min((row for row, _ in component))
    max_row = max((row for row, _ in component))
    layer_height = max_row - min_row + 1
    for row, col in component:
        cleaned_grid[row - layer_height][col] = grid[row][col]

def solve(grid, size=15):
    grid_size = range(size)
    cleaned_grid = [[0] * size for _ in grid]
    processed = set()
    for row in grid_size:
        for col in grid_size:
            if grid[row][col] != 0 and (row, col) not in processed:
                label = grid[row][col]
                component = find_connected_component(grid, row, col, label)
                processed.update(component)
                process_component(component, cleaned_grid, grid)
    return cleaned_grid

def p(grid, size=15):
    return solve(grid, size)
