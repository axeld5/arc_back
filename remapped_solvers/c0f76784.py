def p(grid):
    GRID_SIZE = 12
    visited = [[0] * GRID_SIZE for _ in grid]
    regions = []

    def find_region(start_row, start_col):
        stack = [(start_row, start_col)]
        visited[start_row][start_col] = 1
        region_cells = [(start_row, start_col)]
        is_inside_boundary = True
        while stack:
            current_row, current_col = stack.pop()
            for d_row, d_col in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor_row = current_row + d_row
                neighbor_col = current_col + d_col
                if not (0 <= neighbor_row < GRID_SIZE and 0 <= neighbor_col < GRID_SIZE):
                    is_inside_boundary = False
                    continue
                if grid[neighbor_row][neighbor_col] < 1 and (not visited[neighbor_row][neighbor_col]):
                    visited[neighbor_row][neighbor_col] = 1
                    stack.append((neighbor_row, neighbor_col))
                    region_cells.append((neighbor_row, neighbor_col))
        return region_cells if is_inside_boundary else []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE - 1, -1, -1):
            if grid[row][col] < 1 and (not visited[row][col]):
                region = find_region(row, col)
                if region:
                    regions.append(region)
    regions.sort(key=len, reverse=True)
    for size_index, region in enumerate(regions):
        paint_value = min(8, max(6, int(len(region) ** 0.5 + 0.5 // 1 + 5)))
        for cell in region:
            grid[cell[0]][cell[1]] = paint_value
    return grid
