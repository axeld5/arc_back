from copy import deepcopy

def _transpose(mat):
    return [list(row) for row in zip(*mat)]

def _draw_vertical(grid):
    g = deepcopy(grid)
    R, C = (len(g), len(g[0]))
    cyan = [(r, c) for r in (0, R - 1) for c in range(C) if g[r][c] == 8]
    red = [(r, c) for r in range(R) for c in (0, C - 1) if g[r][c] == 2]
    top_cyan = any((r == 0 for r, _ in cyan))
    step = (1, 0) if top_cyan else (-1, 0)
    anchor_r = 0 if top_cyan else R - 1
    red_on_right = any((c == C - 1 for _, c in red))
    push_c = -1 if red_on_right else 1
    barrier_rows = sorted({r for r, _ in red}, reverse=not top_cyan)
    barrier_rows.append(R if top_cyan else -1)
    shift = 0
    for br in barrier_rows:
        for _, c0 in cyan:
            c = c0 + shift * push_c
            if not 0 <= c < C:
                continue
            r = anchor_r
            while 0 <= r < R and (step[0] > 0 and r < br or (step[0] < 0 and r > br)):
                if g[r][c] == 0:
                    g[r][c] = 8
                r += step[0]
        if br == (R if top_cyan else -1):
            break
        shift += 1
        anchor_r = br
    return g

def p(grid):
    R, C = (len(grid), len(grid[0]))
    cyan = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 8]
    if all((r in (0, R - 1) for r, _ in cyan)):
        return _draw_vertical(grid)
    transposed = _transpose(grid)
    solved = _draw_vertical(transposed)
    return _transpose(solved)
