def p(grid):
    size = len(grid)
    visited = [[0] * size for _ in range(size)]

    def mark_reachable_from_edge(x, y):
        if 0 <= x < size and 0 <= y < size and (not visited[x][y]) and (not grid[x][y]):
            visited[x][y] = 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                mark_reachable_from_edge(x + dx, y + dy)
    for index in range(size):
        mark_reachable_from_edge(index, 0)
        mark_reachable_from_edge(index, size - 1)
        mark_reachable_from_edge(0, index)
        mark_reachable_from_edge(size - 1, index)
    return [[4 * (not grid[row][col] and (not visited[row][col])) or grid[row][col] for col in range(size)] for row in range(size)]
