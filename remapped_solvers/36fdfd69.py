from collections import deque
from itertools import product

def p(grid, range_fn=range, length_fn=len):
    ORIGINAL_COLOR = 2
    MODIFIED_COLOR = 4

    def expand_grid(original_grid, factor=2):
        expanded_grid = []
        for row in original_grid:
            expanded_row = sum(([value] * factor for value in row), [])
            for _ in range_fn(factor):
                expanded_grid.append(expanded_row[:])
        return expanded_grid

    def shrink_grid(expanded_grid, factor=2):
        return [[expanded_grid[i][j] for j in range_fn(0, length_fn(expanded_grid[0]), factor)] for i in range_fn(0, length_fn(expanded_grid), factor)]

    def neighbor_positions(i, j, height, width):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                new_i, new_j = (i + dx, j + dy)
                if 0 <= new_i < height and 0 <= new_j < width:
                    yield (new_i, new_j)

    def find_clusters(grid):
        height, width = (length_fn(grid), length_fn(grid[0]))
        visited = set()
        clusters = []
        for i in range_fn(height):
            for j in range_fn(width):
                if (i, j) in visited or grid[i][j] != ORIGINAL_COLOR:
                    continue
                color = grid[i][j]
                queue = deque([(i, j)])
                cluster_elements = {(i, j)}
                visited.add((i, j))
                while queue:
                    x, y = queue.popleft()
                    for new_x, new_y in neighbor_positions(x, y, height, width):
                        if (new_x, new_y) not in visited and grid[new_x][new_y] == color:
                            visited.add((new_x, new_y))
                            cluster_elements.add((new_x, new_y))
                            queue.append((new_x, new_y))
                clusters.append({'Color': color, 'Elements': cluster_elements})
        return clusters

    def manhattan_distance(cluster1, cluster2):
        return min((abs(i1 - i2) + abs(j1 - j2) for i1, j1 in cluster1 for i2, j2 in cluster2))
    expanded_grid = expand_grid(grid)
    original_color_clusters = [cluster for cluster in find_clusters(expanded_grid) if cluster['Color'] == ORIGINAL_COLOR]
    positions_to_modify = set()
    for cluster1, cluster2 in product(original_color_clusters, repeat=2):
        if manhattan_distance(cluster1['Elements'], cluster2['Elements']) < MODIFIED_COLOR + 1:
            merged_elements = cluster1['Elements'] | cluster2['Elements']
            min_i = min((i for i, _ in merged_elements))
            max_i = max((i for i, _ in merged_elements))
            min_j = min((j for _, j in merged_elements))
            max_j = max((j for _, j in merged_elements))
            for i in range_fn(min_i, max_i + 1):
                for j in range_fn(min_j, max_j + 1):
                    if expanded_grid[i][j] == ORIGINAL_COLOR:
                        continue
                    if (i, j) not in merged_elements:
                        positions_to_modify.add((i, j))
    for i, j in positions_to_modify:
        if expanded_grid[i][j] != ORIGINAL_COLOR:
            expanded_grid[i][j] = MODIFIED_COLOR
    return shrink_grid(expanded_grid)
