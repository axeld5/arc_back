from collections import deque, Counter

def p(I):
    H, W = (len(I), len(I[0]))
    flat = [v for row in I for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    colors = sorted({v for v in flat if v != bg})
    if not colors:
        return [[bg]]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W
    N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    seen = [[False] * W for _ in range(H)]
    comps_by_color = {c: [] for c in colors}
    for r in range(H):
        for c in range(W):
            col = I[r][c]
            if col == bg or seen[r][c]:
                continue
            q = deque([(r, c)])
            seen[r][c] = True
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for dr, dc in N4:
                    nr, nc = (rr + dr, cc + dc)
                    if inb(nr, nc) and (not seen[nr][nc]) and (I[nr][nc] == col):
                        seen[nr][nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps_by_color[col].append(comp)
    dots = {c: cs[0][0] for c, cs in comps_by_color.items() if sum((len(k) for k in cs)) == 1}
    has_dot = bool(dots)
    S = 2 * len(colors) - 1 if has_dot else 2 * len(colors) + 1
    O = [[bg] * S for _ in range(S)]
    cy = cx = S // 2

    def is_ring(cells):
        Sset = set(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = (min(rs), max(rs))
        c0, c1 = (min(cs), max(cs))
        if r1 - r0 != c1 - c0:
            return (False, None)
        side = r1 - r0 + 1
        border = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
        return (Sset == border, side if Sset == border else None)

    def detect_L(cells):
        Sset = set(cells)
        pivot = None
        for r, c in Sset:
            if ((r, c - 1) in Sset or (r, c + 1) in Sset) and ((r - 1, c) in Sset or (r + 1, c) in Sset):
                pivot = (r, c)
                break
        if pivot is None:
            return None
        pr, pc = pivot
        for r, c in Sset:
            if not (r == pr or c == pc):
                return None

        def deg(rc):
            r, c = rc
            return int((r + 1, c) in Sset) + int((r - 1, c) in Sset) + int((r, c + 1) in Sset) + int((r, c - 1) in Sset)
        ends = [rc for rc in Sset if deg(rc) == 1]
        if len(ends) != 2:
            return None
        e1, e2 = ends
        if e1[0] == pr and e2[1] == pc:
            row_end, col_end = (e1, e2)
        elif e2[0] == pr and e1[1] == pc:
            row_end, col_end = (e2, e1)
        else:
            return None
        len_row = abs(row_end[1] - pc) + 1
        len_col = abs(col_end[0] - pr) + 1
        step_row = (0, 1) if row_end[1] > pc else (0, -1)
        step_col = (1, 0) if col_end[0] > pr else (-1, 0)
        return (pivot, step_row, step_col, len_row, len_col)

    def pivot_same_color_distance(pivot, step, color, own):
        r, c = pivot
        dr, dc = step
        d = 0
        while True:
            r += dr
            c += dc
            d += 1
            if not inb(r, c):
                return None
            if I[r][c] == color and (r, c) not in own:
                return d
    info = {}
    for c in colors:
        comps = comps_by_color[c]
        total = sum((len(k) for k in comps))
        if total == 1:
            info[c] = {'kind': 'dot', 'length': 1, 'distance': 0}
            continue
        best = None
        ring_side = None
        for comp in comps:
            okR, side = is_ring(comp)
            if okR:
                ring_side = max(ring_side or 0, side)
                continue
            det = detect_L(comp)
            if det:
                pivot, step_row, step_col, len_row, len_col = det
                own = set(comp)
                d_row = pivot_same_color_distance(pivot, step_row, c, own)
                d_col = pivot_same_color_distance(pivot, step_col, c, own)
                if d_row is None and d_col is None:
                    candidate = (0 + max(len_row, len_col), 'ring', 3, 0)
                else:
                    if d_col is None or (d_row is not None and d_row <= d_col):
                        dist, Llen = (d_row, len_row)
                    else:
                        dist, Llen = (d_col, len_col)
                    candidate = (dist + Llen, 'L', Llen, dist)
                best = min(best, candidate) if best else candidate
        if best:
            _, kind, Llen, dist = best
            info[c] = {'kind': kind, 'length': Llen, 'distance': dist}
        elif ring_side:
            info[c] = {'kind': 'ring', 'length': ring_side, 'distance': 0}
        else:
            cells = [xy for comp in comps for xy in comp]
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            side = max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            info[c] = {'kind': 'ring', 'length': side, 'distance': 0}

    def sort_key(c):
        k = info[c]['kind']
        is_dot = 0 if k == 'dot' else 1
        score = info[c]['distance'] + info[c]['length']
        type_rank = 0 if k == 'ring' else 1 if k == 'L' else 2
        return (is_dot, score, type_rank, info[c]['length'], c)
    order = sorted(colors, key=sort_key)

    def paint_center(color):
        O[cy][cx] = color

    def paint_ring(radius, color):
        s = radius
        r0, r1 = (cy - s, cy + s)
        c0, c1 = (cx - s, cx + s)
        for c in range(c0, c1 + 1):
            O[r0][c] = color
            O[r1][c] = color
        for r in range(r0, r1 + 1):
            O[r][c0] = color
            O[r][c1] = color

    def paint_L(radius, color, Llen):
        s = radius
        if s <= 0:
            return
        L = max(1, min(Llen, s))
        r0, r1 = (cy - s, cy + s)
        c0, c1 = (cx - s, cx + s)
        for c in range(c0, c0 + L):
            O[r0][c] = color
        for r in range(r0, r0 + L):
            O[r][c0] = color
        for c in range(c1 - L + 1, c1 + 1):
            O[r0][c] = color
        for r in range(r0, r0 + L):
            O[r][c1] = color
        for c in range(c0, c0 + L):
            O[r1][c] = color
        for r in range(r1 - L + 1, r1 + 1):
            O[r][c0] = color
        for c in range(c1 - L + 1, c1 + 1):
            O[r1][c] = color
        for r in range(r1 - L + 1, r1 + 1):
            O[r][c1] = color
    max_radius = (S - 1) // 2
    radius = 0
    center_filled = False
    for c in order:
        k = info[c]['kind']
        if k == 'dot' and (not center_filled):
            paint_center(c)
            center_filled = True
            continue
        if radius + 1 > max_radius:
            break
        radius += 1
        if k == 'ring':
            paint_ring(radius, c)
        elif k == 'L':
            paint_L(radius, c, info[c]['length'])
        else:
            paint_ring(radius, c)
    return O
