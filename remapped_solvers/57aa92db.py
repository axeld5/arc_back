from collections import Counter, deque

def p(grid, Range=range):
    height, width = (len(grid), len(grid[0]))
    most_common_value = Counter((value for row in grid for value in row)).most_common(1)[0][0]
    visited = [[0] * width for _ in Range(height)]

    def get_connected_component(i, j):
        queue = deque([(i, j)])
        visited[i][j] = 1
        component = []
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = (x + dx, y + dy)
                if 0 <= nx < height and 0 <= ny < width and (not visited[nx][ny]) and (grid[nx][ny] != most_common_value):
                    visited[nx][ny] = 1
                    queue.append((nx, ny))
        return component
    connected_components = []
    for i in Range(height):
        for j in Range(width):
            if grid[i][j] != most_common_value and (not visited[i][j]):
                connected_components.append(get_connected_component(i, j))
    component_counter = Counter()
    for component in connected_components:
        for value in {grid[i][j] for i, j in component}:
            if value != most_common_value:
                component_counter[value] += 1
    preferred_value = max((component for component, count in component_counter.items() if count > 1), key=component_counter.get)

    def compute_bounds(component):
        pos_list = [(i, j) for i, j in component if grid[i][j] == preferred_value]
        min_i = min((i for i, _ in pos_list))
        min_j = min((j for _, j in pos_list))
        max_dim = max(max((i for i, _ in pos_list)) - min_i + 1, max((j for _, j in pos_list)) - min_j + 1)
        return (min_i, min_j, max_dim)
    target_component = min((component for component in connected_components if sum((grid[i][j] != preferred_value for i, j in component)) > compute_bounds(component)[2] ** 2), key=lambda component: compute_bounds(component)[2])
    target_i, target_j, target_dim = compute_bounds(target_component)
    updated_positions = [(i - target_i, j - target_j) for i, j in target_component if grid[i][j] != preferred_value]
    result_grid = [row[:] for row in grid]
    for component in connected_components:
        comp_i, comp_j, comp_dim = compute_bounds(component)
        diff_value = next((value for value in {grid[i][j] for i, j in component} if value not in (most_common_value, preferred_value)))
        for di, dj in updated_positions:
            dest_i, dest_j = (comp_i + di * comp_dim, comp_j + dj * comp_dim)
            for i in Range(comp_dim):
                for j in Range(comp_dim):
                    new_i, new_j = (dest_i + i, dest_j + j)
                    if 0 <= new_i < height and 0 <= new_j < width:
                        result_grid[new_i][new_j] = diff_value
    return result_grid
