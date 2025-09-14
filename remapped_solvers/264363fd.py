from collections import *

def p(g):
    H, W = (len(g), len(g[0]))
    D = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def most_common(x):
        return Counter((v for r in x for v in r)).most_common(1)[0][0]

    def find_components(x, bg):
        visited = set()
        objects = []
        for r in range(H):
            for c in range(W):
                if (r, c) in visited or x[r][c] == bg:
                    continue
                queue = deque([(r, c)])
                visited.add((r, c))
                current = []
                while queue:
                    i, j = queue.popleft()
                    current.append((x[i][j], i, j))
                    for di, dj in D:
                        ni, nj = (i + di, j + dj)
                        if in_bounds(ni, nj) and (ni, nj) not in visited and (x[ni][nj] != bg):
                            visited.add((ni, nj))
                            queue.append((ni, nj))
                objects.append(current)
        return objects

    def get_bounds(component):
        return (min((r for _, r, _ in component)), max((r for _, r, _ in component)), min((c for _, _, c in component)), max((c for _, _, c in component)))

    def get_center(r0, r1, c0, c1):
        return (r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2)

    def copy_grid(x):
        return [list(r) for r in x]
    BG = most_common(g)
    objects = find_components(g, BG)
    smallest = min(objects, key=len)
    sr0, sr1, sc0, sc1 = get_bounds(smallest)
    height, width = (sr1 - sr0 + 1, sc1 - sc0 + 1)
    center_r, center_c = get_center(sr0, sr1, sc0, sc1)
    center_color = g[center_r][center_c]
    direction = (-1, 0) if height == 5 else (0, 1)
    next_r, next_c = (center_r + direction[0], center_c + direction[1])
    color_line = g[next_r][next_c]
    g2 = copy_grid(g)
    for _, r, c in smallest:
        g2[r][c] = BG
    BG2 = most_common(g2)
    background_cells = {(r, c) for r in range(H) for c in range(W) if g2[r][c] == BG2}
    occurrences = [(r, c) for r in range(H) for c in range(W) if g2[r][c] == center_color]
    for obj in find_components(g2, BG2):
        r0, r1, c0, c1 = get_bounds(obj)
        locations = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if g2[r][c] == center_color]
        for r, c in locations:
            if height == 5:
                for i in range(r0, r1 + 1):
                    g2[i][c] = color_line
            else:
                for j in range(c0, c1 + 1):
                    g2[r][j] = color_line
            if width == 5:
                for j in range(c0, c1 + 1):
                    g2[r][j] = color_line
            else:
                for i in range(r0, r1 + 1):
                    g2[i][c] = color_line
    normalized = [(v, r - sr0, c - sc0) for v, r, c in smallest]
    dx = -(1 + (height == 5))
    dy = -(1 + (width == 5))
    for r, c in occurrences:
        base_r, base_c = (r + dx, c + dy)
        for v, rr, cc in normalized:
            nr, nc = (base_r + rr, base_c + cc)
            if in_bounds(nr, nc):
                g2[nr][nc] = v
    for r, c in background_cells:
        g2[r][c] = BG2
    return [list(r) for r in g2]
