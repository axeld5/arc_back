def p(grid):
    H, W = (len(grid), len(grid[0]))
    yellow = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 4]
    r0, r1 = (min((r for r, _ in yellow)), max((r for r, _ in yellow)))
    c0, c1 = (min((c for _, c in yellow)), max((c for _, c in yellow)))
    tall, wide = (r1 - r0 + 1, c1 - c0 + 1)
    small_pixels = []
    for r in range(H):
        for c in range(W):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            val = grid[r][c]
            if val != 0 and val != 4:
                small_pixels.append((r, c, val))
    rs_min = min((r for r, _, _ in small_pixels))
    cs_min = min((c for _, c, _ in small_pixels))
    rs_max = max((r for r, _, _ in small_pixels))
    cs_max = max((c for _, c, _ in small_pixels))
    h_small = rs_max - rs_min + 1
    w_small = cs_max - cs_min + 1
    small = {(r - rs_min, c - cs_min): col for r, c, col in small_pixels}
    inside = [(r - r0, c - c0, grid[r][c]) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r][c] not in (0, 4)]
    SOL = None
    for m in (2, 3, 4):
        h_big, w_big = (h_small * m, w_small * m)
        for dy in range(0, tall - h_big + 1):
            for dx in range(0, wide - w_big + 1):
                ok = True
                for ir, ic, col in inside:
                    rr, cc = (ir - dy, ic - dx)
                    if not (0 <= rr < h_big and 0 <= cc < w_big):
                        ok = False
                        break
                    rs, cs = (rr // m, cc // m)
                    if small.get((rs, cs), None) != col:
                        ok = False
                        break
                if ok:
                    SOL = (m, dy, dx)
                    break
            if SOL:
                break
        if SOL:
            break
    if SOL is None:
        raise ValueError('No consistent placement found')
    m, dy, dx = SOL
    out = [[0] * wide for _ in range(tall)]
    out[0][0] = out[0][wide - 1] = out[tall - 1][0] = out[tall - 1][wide - 1] = 4
    for (rs, cs), col in small.items():
        for dr in range(m):
            for dc in range(m):
                r = dy + rs * m + dr
                c = dx + cs * m + dc
                out[r][c] = col
    return out
