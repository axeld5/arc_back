from collections import Counter
from statistics import median

def p(grid):
    H, W = (len(grid), len(grid[0]))
    BG = 0
    counts = Counter()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != BG:
                counts[grid[r][c]] += 1
    colors = [c for c in counts if c != BG]

    def is_filled_rectangle(color):
        rows = [r for r in range(H) for c in range(W) if grid[r][c] == color]
        if not rows:
            return (False, None)
        cols = [c for r in range(H) for c in range(W) if grid[r][c] == color]
        rmin, rmax = (min(rows), max(rows))
        cmin, cmax = (min(cols), max(cols))
        area = (rmax - rmin + 1) * (cmax - cmin + 1)
        if area != counts[color]:
            return (False, None)
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if grid[r][c] != color:
                    return (False, None)
        return (True, (rmin, rmax, cmin, cmax))
    rect_color = None
    rect_bbox = None
    for col in colors:
        ok, bbox = is_filled_rectangle(col)
        if ok:
            rect_color, rect_bbox = (col, bbox)
            break
    if rect_color is None:
        return [row[:] for row in grid]
    out = [row[:] for row in grid]
    rect_rows = set()
    rmin, rmax, cmin, cmax = rect_bbox
    for r in range(rmin, rmax + 1):
        rect_rows.add(r)
        for c in range(cmin, cmax + 1):
            out[r][c] = BG
    axes = []
    for r in range(H):
        if r in rect_rows:
            continue
        cols = [c for c in range(W) if out[r][c] != BG]
        if cols:
            axes.append((min(cols) + max(cols)) / 2.0)
    axis = median(axes) if axes else (W - 1) / 2.0
    res = [row[:] for row in out]

    def mirror_col(c, ax):
        return int(round(2 * ax - c))
    for r in range(H):
        for c in range(W):
            val = res[r][c]
            if val == BG:
                continue
            mc = mirror_col(c, axis)
            if 0 <= mc < W and res[r][mc] == BG:
                res[r][mc] = val
    return res
