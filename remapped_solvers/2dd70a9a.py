from collections import deque
from copy import deepcopy
UP, DOWN, LEFT, RIGHT = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}
GREEN, RED, CYAN, BG = (3, 2, 8, 0)

def inside(r, c, h, w):
    return 0 <= r < h and 0 <= c < w

def p(grid):
    H, W = (len(grid), len(grid[0]))
    g0 = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == GREEN}
    reds = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == RED]
    if len(reds) != 2:
        raise ValueError('Puzzle must have exactly two red cells')
    (r1, c1), (r2, c2) = reds
    vertical = c1 == c2

    def red_arrival_ok(dr, dc):
        return vertical and dc == 0 and dr or (not vertical and dr == 0 and dc)
    heads = []
    for r, c in g0:
        deg = sum(((r + dr, c + dc) in g0 for dr, dc in (UP, DOWN, LEFT, RIGHT)))
        if deg <= 1:
            for dr, dc in (UP, DOWN, LEFT, RIGHT):
                if (r + dr, c + dc) in g0:
                    heads.append((r, c, dr, dc))
                    break
            else:
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    heads.append((r, c, dr, dc))
    q = deque()
    seen = set()
    for r, c, dr, dc in heads:
        q.append((r, c, dr, dc, [(r, c)], {(r, c)}))
        seen.add((r, c, dr, dc))
    while q:
        r, c, dr, dc, path, path_set = q.popleft()
        nr, nc = (r + dr, c + dc)
        if not inside(nr, nc, H, W):
            continue
        cell = grid[nr][nc]
        if cell == CYAN:
            for pdr, pdc in PERP[dr, dc]:
                key = (r, c, pdr, pdc)
                if key not in seen:
                    seen.add(key)
                    q.append((r, c, pdr, pdc, path, path_set))
            continue
        if cell == BG or (cell == GREEN and (nr, nc) in g0):
            if (nr, nc) not in path_set:
                key = (nr, nc, dr, dc)
                if key not in seen:
                    seen.add(key)
                    q.append((nr, nc, dr, dc, path + [(nr, nc)], path_set | {(nr, nc)}))
            continue
        if cell == RED and red_arrival_ok(dr, dc):
            return _paint(grid, path + [(nr, nc)])
    raise RuntimeError('No valid path exists')

def _paint(pic, path):
    out = deepcopy(pic)
    for r, c in path:
        if out[r][c] == BG:
            out[r][c] = GREEN
    return out
