from collections import Counter, deque

def p(grid):

    def H(g):
        return len(g)

    def W(g):
        return 0 if not g else len(g[0])

    def mostcolour(g):
        return Counter((v for r in g for v in r)).most_common(1)[0][0]

    def neigh4(i, j):
        return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
    h, w = (H(grid), W(grid))
    g = [row[:] for row in grid]
    reds = [(i, j) for i in range(h) for j in range(w) if g[i][j] == 2]
    if not reds:
        return g
    rmin = min((i for i, _ in reds))
    rmax = max((i for i, _ in reds))
    cmin, cmax = (0, w - 1)
    seen = [[False] * w for _ in range(h)]
    holes = []
    for i in range(rmin, rmax + 1):
        for j in range(cmin, cmax + 1):
            if g[i][j] != 0 or seen[i][j]:
                continue
            q = deque([(i, j)])
            seen[i][j] = True
            comp = set()
            while q:
                x, y = q.popleft()
                comp.add((x, y))
                for nx, ny in neigh4(x, y):
                    if rmin <= nx <= rmax and cmin <= ny <= cmax and (not seen[nx][ny]) and (g[nx][ny] == 0):
                        seen[nx][ny] = True
                        q.append((nx, ny))
            holes.append(comp)
    holes_band = set()
    for comp in holes:
        holes_band |= comp
    seen5 = [[False] * w for _ in range(h)]
    pieces = []
    for i in range(rmax + 1, h):
        for j in range(w):
            if g[i][j] != 5 or seen5[i][j]:
                continue
            q = deque([(i, j)])
            seen5[i][j] = True
            comp = set()
            while q:
                x, y = q.popleft()
                comp.add((x, y))
                for nx, ny in neigh4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and (not seen5[nx][ny]) and (g[nx][ny] == 5):
                        seen5[nx][ny] = True
                        q.append((nx, ny))
            pieces.append(comp)
    first_rows = min(5, h)
    idx5rows = {(i, j) for i in range(first_rows) for j in range(w)}
    idx0 = {(i, j) for i in range(first_rows) for j in range(w) if grid[i][j] == 0}
    idx2 = {(i, j) for i in range(first_rows) for j in range(w) if grid[i][j] == 2}
    bg = mostcolour(g)
    for comp in pieces:
        for i, j in comp:
            g[i][j] = bg
    norm_pieces = []
    for comp in pieces:
        t = min((i for i, _ in comp))
        l = min((j for _, j in comp))
        norm_pieces.append({(i - t, j - l) for i, j in comp})
    candidates = []
    for hole in holes:
        hole_set = set(hole)
        hcells = list(hole)
        seen_T = set()
        cand = []
        for pidx, shape in enumerate(norm_pieces):
            for sr, sc in shape:
                for hr, hc in hcells:
                    dr, dc = (hr - sr, hc - sc)
                    T = {(dr + r, dc + c) for r, c in shape}
                    if min((r for r, _ in T)) < rmin:
                        continue
                    if not all((0 <= r < h and 0 <= c < w for r, c in T)):
                        continue
                    Tin = {(r, c) for r, c in T if rmin <= r <= rmax and cmin <= c <= cmax}
                    if Tin == hole_set:
                        key = (pidx, frozenset(T))
                        if key not in seen_T:
                            seen_T.add(key)
                            cand.append((pidx, T))
        candidates.append(cand)
    used_piece = [False] * len(norm_pieces)
    placement = [None] * len(holes)
    order = sorted(range(len(holes)), key=lambda k: len(candidates[k]))
    painted = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        hk = order[pos]
        for pidx, T in candidates[hk]:
            if used_piece[pidx] or painted & T:
                continue
            used_piece[pidx] = True
            placement[hk] = (pidx, T)
            painted.update(T)
            if dfs(pos + 1):
                return True
            painted.difference_update(T)
            placement[hk] = None
            used_piece[pidx] = False
        return False
    need_fallback = len(holes) > 0 and any((len(c) == 0 for c in candidates)) or (len(holes) > 0 and (not dfs(0)))
    if need_fallback:

        def bbox(p):
            r = [x for x, _ in p]
            c = [y for _, y in p]
            return (min(r), max(r), min(c), max(c))

        def delta(p):
            if not p:
                return set()
            t, b, l, r = bbox(p)
            return {(i, j) for i in range(t, b + 1) for j in range(l, r + 1)} - p

        def outbox(p):
            if not p:
                return set()
            t, b, l, r = bbox(p)
            t, b, l, r = (t - 1, b + 1, l - 1, r + 1)
            box = set()
            for i in range(t, b + 1):
                box.add((i, l))
                box.add((i, r))
            for j in range(l, r + 1):
                box.add((t, j))
                box.add((b, j))
            return box

        def uppermost(p):
            return min((i for i, _ in p))

        def shift(p, d):
            di, dj = d
            return {(i + di, j + dj) for i, j in p}
        drawn = set()
        covered = set()

        def is_red_cell(rc):
            r, c = rc
            return grid[r][c] == 2
        for shape in norm_pieces:
            best_s, best_patch = (-10 ** 9, None)
            for i in range(first_rows):
                for j in range(w):
                    patch = shift(shape, (i, j))
                    if not all((0 <= r < h and 0 <= c < w for r, c in patch)):
                        continue
                    if any((is_red_cell((r, c)) for r, c in patch)):
                        continue
                    if patch & drawn:
                        continue
                    patch_in_band = {(r, c) for r, c in patch if rmin <= r <= rmax and cmin <= c <= cmax}
                    newly_covering = patch_in_band & holes_band - covered
                    hole_gain = len(newly_covering)
                    v22 = 10 * len(patch & idx5rows)
                    v18 = 10 * len(patch & idx2)
                    v24 = v22 - v18
                    v21 = len(outbox(patch) & idx0)
                    v25 = v24 - v21
                    v20 = 5 * uppermost(patch)
                    v26 = v25 - v20
                    v19 = len(delta(patch) & idx0)
                    s = 1000 * hole_gain + (v26 - v19)
                    if s > best_s:
                        best_s, best_patch = (s, patch)
            if best_patch:
                drawn |= best_patch
                covered |= {(r, c) for r, c in best_patch if rmin <= r <= rmax and cmin <= c <= cmax} & holes_band
        for i, j in drawn:
            g[i][j] = 1
        return g
    for itm in placement:
        if itm is None:
            continue
        _, T = itm
        for r, c in T:
            g[r][c] = 1
    return g
