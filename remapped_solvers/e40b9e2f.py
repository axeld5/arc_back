def p(grid):
    H = W = 10
    bg = 0
    out = [row[:] for row in grid]
    F = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not F:
        return out
    rmin = min((r for r, _ in F))
    rmax = max((r for r, _ in F))
    cmin = min((c for _, c in F))
    cmax = max((c for _, c in F))
    rb = (rmin + rmax) / 2.0
    cb = (cmin + cmax) / 2.0
    best = None
    for k in (3, 2):
        for a in range(0, H - k + 1):
            for b in range(0, W - k + 1):
                cnt = sum((grid[r][c] != bg for r in range(a, a + k) for c in range(b, b + k)))
                density = cnt / float(k * k)
                cy = a + k / 2.0
                cx = b + k / 2.0
                dist2 = (cy - rb) ** 2 + (cx - cb) ** 2
                key = (density, k, -dist2, -a, -b)
                if best is None or key > best[0]:
                    best = (key, a, b, k, cy, cx)
    _, a, b, k, y0, x0 = best
    hub = {(r, c) for r in range(a, a + k) for c in range(b, b + k)}

    def rot90(r, c):
        y, x = (r + 0.5, c + 0.5)
        y2 = y0 - (x - x0)
        x2 = x0 + (y - y0)
        return (int(round(y2 - 0.5)), int(round(x2 - 0.5)))
    for r, c in F:
        if (r, c) in hub:
            continue
        col = grid[r][c]
        r2, c2 = rot90(r, c)
        if 0 <= r2 < H and 0 <= c2 < W and (out[r2][c2] == bg):
            out[r2][c2] = col
        r3, c3 = rot90(r2, c2)
        if 0 <= r3 < H and 0 <= c3 < W and (out[r3][c3] == bg):
            out[r3][c3] = col
        r4, c4 = rot90(r3, c3)
        if 0 <= r4 < H and 0 <= c4 < W and (out[r4][c4] == bg):
            out[r4][c4] = col
    return out
