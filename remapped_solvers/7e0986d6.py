from collections import Counter, deque

def dfs_mark_largest_region(grid, start_row, start_col, mark_value, visited):
    H, W = (len(grid), len(grid[0]))
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    min_row = max_row = start_row
    min_col = max_col = start_col
    stack = deque([(start_row, start_col)])
    visited[start_row][start_col] = True
    while stack:
        r, c = stack.pop()
        min_row, max_row = (min(min_row, r), max(max_row, r))
        min_col, max_col = (min(min_col, c), max(max_col, c))
        for dr, dc in directions:
            nr, nc = (r + dr, c + dc)
            if 0 <= nr < H and 0 <= nc < W and (not visited[nr][nc]) and (grid[nr][nc] == mark_value):
                visited[nr][nc] = True
                stack.append((nr, nc))
    return (min_row, max_row, min_col, max_col)

def mark_invalid_isolated_pixels(output_grid, largest_value):
    H, W = (len(output_grid), len(output_grid[0]))
    result_grid = [row[:] for row in output_grid]
    for r in range(H):
        for c in range(W):
            if output_grid[r][c] == largest_value:
                is_isolated = True
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = (r + dr, c + dc)
                        if (dr != 0 or dc != 0) and (0 <= nr < H and 0 <= nc < W) and (output_grid[nr][nc] != 0):
                            is_isolated = False
                            break
                    if not is_isolated:
                        break
                if is_isolated:
                    result_grid[r][c] = 0
    return result_grid

def p(grid):
    H, W = (len(grid), len(grid[0]))
    value_counter = Counter((v for row in grid for v in row if v))
    (most_common_value, _), (second_most_common_value, _) = value_counter.most_common(2)
    visited = [[False] * W for _ in range(H)]
    regions_bounds = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == most_common_value and (not visited[i][j]):
                region_bounds = dfs_mark_largest_region(grid, i, j, most_common_value, visited)
                regions_bounds.append(region_bounds)
    output_grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == most_common_value:
                output_grid[r][c] = most_common_value
            elif v == second_most_common_value:
                for min_row, max_row, min_col, max_col in regions_bounds:
                    if min_row <= r <= max_row and min_col <= c <= max_col:
                        output_grid[r][c] = most_common_value
                        break
    cleaned_grid = mark_invalid_isolated_pixels(output_grid, most_common_value)
    return cleaned_grid
