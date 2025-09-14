from collections import Counter, deque
BLACK, YELLOW, GREEN = (0, 4, 3)

def p(grid):
    height, width = (len(grid), len(grid[0]))
    output_grid = [row[:] for row in grid]
    flat_cells = [cell for row in grid for cell in row]
    least_frequent_color = next((color for color, _ in Counter(flat_cells).items() if color not in (BLACK, YELLOW, GREEN)))
    columns_to_change = {col for col in range(width) if sum((grid[row][col] == least_frequent_color for row in range(height))) > height // 2}
    rows_to_change = {row for row in range(height) if sum((grid[row][col] == least_frequent_color for col in range(width))) > width // 2}
    for row in rows_to_change:
        for col in range(width):
            if output_grid[row][col] == BLACK:
                output_grid[row][col] = YELLOW
    for col in columns_to_change:
        for row in range(height):
            if output_grid[row][col] == BLACK:
                output_grid[row][col] = YELLOW
    visited = [[0] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for row in range(height):
        for col in range(width):
            if output_grid[row][col] != BLACK or visited[row][col]:
                continue
            queue = deque([(row, col)])
            region = []
            borders_with_yellow = False
            while queue:
                x, y = queue.popleft()
                if visited[x][y]:
                    continue
                visited[x][y] = 1
                region.append((x, y))
                for dx, dy in directions:
                    nx, ny = (x + dx, y + dy)
                    if 0 <= nx < height and 0 <= ny < width:
                        if output_grid[nx][ny] == BLACK and (not visited[nx][ny]):
                            queue.append((nx, ny))
                        elif output_grid[nx][ny] == YELLOW:
                            borders_with_yellow = True
            for x, y in region:
                output_grid[x][y] = YELLOW if borders_with_yellow else GREEN
    return output_grid
