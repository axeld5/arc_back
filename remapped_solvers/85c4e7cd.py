def p(grid):
    height = len(grid)
    visited = [[False] * height for _ in range(height)]
    components = []

    def find_component(start_i, start_j, color):
        queue = [(start_i, start_j)]
        component_cells = []
        visited[start_i][start_j] = True
        while queue:
            x, y = queue.pop()
            component_cells.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = (x + dx, y + dy)
                if 0 <= new_x < height and 0 <= new_y < height and (not visited[new_x][new_y]) and (grid[new_x][new_y] == color):
                    visited[new_x][new_y] = True
                    queue.append((new_x, new_y))
        return component_cells
    for i in range(height):
        for j in range(height):
            if not visited[i][j]:
                color = grid[i][j]
                cells = find_component(i, j, color)
                components.append({'color': color, 'cells': cells})
    smallest_components = sorted(components, key=lambda x: len(x['cells']))
    largest_components = sorted(components, key=lambda x: len(x['cells']), reverse=True)
    output = [row[:] for row in grid]
    for small_comp, large_comp in zip(smallest_components, largest_components):
        new_color = large_comp['color']
        for i, j in small_comp['cells']:
            output[i][j] = new_color
    return output
