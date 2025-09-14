def p(grid):

    def is_out_of_bounds(x, y, max_x, max_y):
        return x < 0 or y < 0 or x >= max_x or (y >= max_y)

    def bfs_mark_and_check_rectangular(start_x, start_y, component_value):
        queue = [(start_x, start_y)]
        visited[start_x][start_y] = True
        min_row, max_row = (start_x, start_x)
        min_col, max_col = (start_y, start_y)
        component_size = 0
        while queue:
            x, y = queue.pop()
            component_size += 1
            min_row = min(min_row, x)
            max_row = max(max_row, x)
            min_col = min(min_col, y)
            max_col = max(max_col, y)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = (x + dx, y + dy)
                if not is_out_of_bounds(nx, ny, rows_count, cols_count) and (not visited[nx][ny]) and (grid[nx][ny] == component_value):
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        expected_size = (max_row - min_row + 1) * (max_col - min_col + 1)
        return component_size == expected_size
    rows_count = len(grid)
    cols_count = len(grid[0])
    visited = [[False] * cols_count for _ in range(rows_count)]
    for row in range(rows_count):
        for col in range(cols_count):
            if grid[row][col] == 0 or visited[row][col]:
                continue
            component_value = grid[row][col]
            if not bfs_mark_and_check_rectangular(row, col, component_value):
                return [[component_value]]
    return [[0]]
