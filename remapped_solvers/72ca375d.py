def is_symmetric_2d_list(subgrid):
    return subgrid == [row[::-1] for row in subgrid]

def get_neighbors(x, y):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return [(x + dx, y + dy) for dx, dy in directions]

def bfs_collect_region(grid, visited, start_x, start_y):
    queue = [(start_x, start_y)]
    connected_region = []
    visited[start_x][start_y] = True
    cell_value = grid[start_x][start_y]
    while queue:
        x, y = queue.pop()
        connected_region.append((x, y))
        for neighbor_x, neighbor_y in get_neighbors(x, y):
            if 0 <= neighbor_x < 10 and 0 <= neighbor_y < 10 and (not visited[neighbor_x][neighbor_y]) and (grid[neighbor_x][neighbor_y] == cell_value):
                visited[neighbor_x][neighbor_y] = True
                queue.append((neighbor_x, neighbor_y))
    return connected_region

def extract_subgrid(grid, region):
    min_row = min((r for r, _ in region))
    max_row = max((r for r, _ in region))
    min_col = min((c for _, c in region))
    max_col = max((c for _, c in region))
    return [list(grid[r][min_col:max_col + 1]) for r in range(min_row, max_row + 1)]

def p(grid, range_obj=range(10)):
    visited = [[False] * 10 for _ in range_obj]
    regions = []
    for i in range_obj:
        for j in range_obj:
            if not visited[i][j] and grid[i][j] != 0:
                region = bfs_collect_region(grid, visited, i, j)
                regions.append(region)
    for region in regions:
        subgrid = extract_subgrid(grid, region)
        if is_symmetric_2d_list(subgrid):
            return subgrid
    return None
