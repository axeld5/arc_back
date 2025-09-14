from collections import deque, Counter

def p(grid, grid_size=10):

    def get_neighbors(row, col):
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                if x or y:
                    new_row, new_col = (row + x, col + y)
                    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                        yield (new_row, new_col)

    def find_all_components():
        visited = set()
        components = []
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in visited or grid[i][j] == 0:
                    continue
                value = grid[i][j]
                queue = deque([(i, j)])
                component = [(i, j)]
                visited.add((i, j))
                while queue:
                    r, c = queue.popleft()
                    for u, w in get_neighbors(r, c):
                        if (u, w) not in visited and grid[u][w] == value:
                            visited.add((u, w))
                            queue.append((u, w))
                            component.append((u, w))
                components.append(component)
        return components

    def normalize_component(component):
        min_row = min((r for r, _ in component))
        min_col = min((c for _, c in component))
        return tuple(sorted(((r - min_row, c - min_col) for r, c in component)))
    all_components = find_all_components()
    normalized_shapes = [normalize_component(comp) for comp in all_components]
    shape_counter = Counter(normalized_shapes)
    target_shape = min(normalized_shapes, key=lambda s: (shape_counter[s], normalized_shapes.index(s)))
    target_component = all_components[normalized_shapes.index(target_shape)]
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r][c] == 8:
                grid[r][c] = 1
    for r, c in target_component:
        grid[r][c] = 2
    return grid
