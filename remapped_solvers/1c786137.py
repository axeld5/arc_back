def extract_interior(grid, rect):
    r1, c1, r2, c2 = rect
    return [row[c1 + 1:c2] for row in grid[r1 + 1:r2]]

def p(grid):
    m, n = (len(grid), len(grid[0]))
    right = [[0] * n for _ in range(m)]
    down = [[0] * n for _ in range(m)]
    for r in reversed(range(m)):
        for c in reversed(range(n)):
            right[r][c] = 1 + (right[r][c + 1] if c + 1 < n and grid[r][c] == grid[r][c + 1] else 0)
            down[r][c] = 1 + (down[r + 1][c] if r + 1 < m and grid[r][c] == grid[r + 1][c] else 0)
    best_area, best_rect = (-1, None)
    for r1 in range(m):
        for c1 in range(n):
            v = grid[r1][c1]
            if v == 0:
                continue
            max_w = right[r1][c1]
            for w in range(2, max_w + 1):
                c2 = c1 + w - 1
                max_h = min(down[r1][c1], down[r1][c2])
                for h in range(2, max_h + 1):
                    r2 = r1 + h - 1
                    if right[r2][c1] < w:
                        continue
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, c1, r2, c2)
    if not best_rect:
        return [[0]]
    r1, c1, r2, c2 = best_rect
    interior = [row[c1 + 1:c2] for row in grid[r1 + 1:r2]]
    return interior
