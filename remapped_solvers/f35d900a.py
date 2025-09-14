def p(grid, range_func=range):

    def get_border_coordinates(component):
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        min_row, max_row = (min(rows) - 1, max(rows) + 1)
        min_col, max_col = (min(cols) - 1, max(cols) + 1)
        border_coords = set()
        for row in range_func(min_row, max_row + 1):
            border_coords.add((row, min_col))
            border_coords.add((row, max_col))
        for col in range_func(min_col, max_col + 1):
            border_coords.add((min_row, col))
            border_coords.add((max_row, col))
        return border_coords

    def calculate_distance_to_region(p):
        p_row, p_col = p
        return min((abs(p_row - r) + abs(p_col - c) for r, c in region_coords))
    num_rows, num_cols = (len(grid), len(grid[0]))
    unique_values = {value for row in grid for value in row if value != 0}
    visited = set()
    components = []
    for i in range_func(num_rows):
        for j in range_func(num_cols):
            if grid[i][j] == 0 or (i, j) in visited:
                continue
            current_value = grid[i][j]
            stack = [(i, j)]
            component = set()
            while stack:
                x, y = stack.pop()
                if (x, y) in visited or grid[x][y] != current_value:
                    continue
                visited.add((x, y))
                component.add((x, y))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = (x + dx, y + dy)
                    if 0 <= nx < num_rows and 0 <= ny < num_cols and (grid[nx][ny] == current_value):
                        stack.append((nx, ny))
            components.append({'value': current_value, 'coordinates': component})
    modified_grid = [list(row) for row in grid]
    for component in components:
        border_value = next((val for val in unique_values if val != component['value']))
        border_coords = get_border_coordinates(component['coordinates'])
        for row, col in border_coords:
            if 0 <= row < num_rows and 0 <= col < num_cols:
                modified_grid[row][col] = border_value
    region_coords = {coord for component in components for coord in component['coordinates']}
    region_rows = [r for r, _ in region_coords]
    region_cols = [c for _, c in region_coords]
    min_r, max_r, min_c, max_c = (min(region_rows), max(region_rows), min(region_cols), max(region_cols))
    boundary_coords = {(r, min_c) for r in range_func(min_r, max_r + 1)} | {(r, max_c) for r in range_func(min_r, max_r + 1)} | {(min_r, c) for c in range_func(min_c, max_c + 1)} | {(max_r, c) for c in range_func(min_c, max_c + 1)}
    outer_boundary_coords = boundary_coords - region_coords
    even_distance_boundary = {p for p in outer_boundary_coords if calculate_distance_to_region(p) % 2 == 0}
    for row, col in even_distance_boundary:
        if 0 <= row < num_rows and 0 <= col < num_cols:
            modified_grid[row][col] = 5
    return modified_grid
