def flood_fill(grid, start_points, rows, cols):
    while start_points:
        x, y = start_points.pop()
        if grid[x][y] < 1:
            grid[x][y] = 3
            start_points.extend(((nx, ny) for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if 0 <= nx < rows and 0 <= ny < cols and (grid[nx][ny] < 1)))

def mark_internal_areas(grid):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] < 1:
                grid[x][y] = 2

def p(grid):
    if not grid or not grid[0]:
        return grid
    num_rows, num_cols = (len(grid), len(grid[0]))
    edge_points = [(i, j) for i in range(num_rows) for j in range(num_cols) if (i in (0, num_rows - 1) or j in (0, num_cols - 1)) and grid[i][j] < 1]
    flood_fill(grid, edge_points, num_rows, num_cols)
    mark_internal_areas(grid)
    return grid
