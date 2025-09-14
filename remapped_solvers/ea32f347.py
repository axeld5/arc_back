def p(grid):

    def find_clusters(grid):
        rows, cols = (len(grid), len(grid[0]))
        clusters = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 5:
                    cluster = explore_cluster(grid, r, c)
                    clusters.append(cluster)
        return clusters

    def explore_cluster(grid, start_row, start_col):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        stack = [(start_row, start_col)]
        cluster = []
        while stack:
            row, col = stack.pop()
            if grid[row][col] == 5:
                grid[row][col] = 0
                cluster.append((row, col))
                for dr, dc in directions:
                    new_row, new_col = (row + dr, col + dc)
                    if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                        if grid[new_row][new_col] == 5:
                            stack.append((new_row, new_col))
        return cluster

    def assign_values_to_clusters(grid, clusters):
        replacement_values = (2, 4, 1)
        for cluster, value in zip(sorted(clusters, key=len), replacement_values):
            for row, col in cluster:
                grid[row][col] = value
    clusters = find_clusters(grid)
    assign_values_to_clusters(grid, clusters)
    return grid
