def p(grid):
    n = len(grid)
    seen = set()
    transformed_grid = [row[:] for row in grid]

    def find_cluster(x, y):
        if (x, y) in seen or not (0 <= x < n and 0 <= y < n) or grid[x][y] != 1:
            return []
        seen.add((x, y))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        cluster = [(x, y)]
        for dx, dy in directions:
            cluster.extend(find_cluster(x + dx, y + dy))
        return cluster
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1 and (i, j) not in seen:
                cluster = find_cluster(i, j)
                min_x = min((c[0] for c in cluster))
                max_x = max((c[0] for c in cluster))
                min_y = min((c[1] for c in cluster))
                max_y = max((c[1] for c in cluster))
                if len(cluster) == 2 * (max_x - min_x + max_y - min_y) and max_x > min_x and (max_y > min_y) and any((grid[x][y] == 0 for x in range(min_x + 1, max_x) for y in range(min_y + 1, max_y))):
                    for x, y in cluster:
                        transformed_grid[x][y] = 3
    return transformed_grid
