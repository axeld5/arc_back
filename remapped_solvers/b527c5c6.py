from collections import *

def p(g):
    H, W = (len(g), len(g[0]))

    def i(r, c):
        return 0 <= r < H and 0 <= c < W

    def n(r, c):
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    bg = Counter([v for row in g for v in row]).most_common(1)[0][0]
    G = [list(r) for r in g]
    s, o = (set(), [])
    for r in range(H):
        for c in range(W):
            if (r, c) in s or G[r][c] == bg:
                continue
            q, blk = (deque([(r, c)]), [])
            while q:
                x, y = q.popleft()
                if (x, y) in s or G[x][y] == bg:
                    continue
                s.add((x, y))
                blk.append((x, y))
                for nx, ny in n(x, y):
                    if i(nx, ny):
                        q.append((nx, ny))
            o.append(blk)

    def bb(c):
        rs, cs = zip(*c)
        return (min(rs), max(rs), min(cs), max(cs))

    def ct(c):
        t, btm, l, r = bb(c)
        return (t + (btm - t) // 2, l + (r - l) // 2)

    def sh(st, d):
        di, dj = d
        if di == dj == 0:
            return {st}
        out = set()
        x, y = st
        while i(x, y):
            out.add((x, y))
            x += di
            y += dj
        return out
    ml, li = (set(), [])
    for blk in o:
        ts = [(x, y) for x, y in blk if g[x][y] == 2]
        if not ts:
            continue
        t2, b2, l2, r2 = bb(ts)
        t, btm, l, r = bb(blk)
        di = (-1 if t2 == t else 0) + (1 if b2 == btm else 0)
        dj = (-1 if l2 == l else 0) + (1 if r2 == r else 0)
        ln = sh(ct(ts), (di, dj))
        ml |= ln
        li.append((blk, ln, (di, dj)))
    for x, y in ml:
        G[x][y] = 2
    st = set()
    for blk, ln, (di, dj) in li:
        t, btm, l, r = bb(blk)
        h, w = (btm - t + 1, r - l + 1)
        m = min(h, w)
        of = range(-(m - 1), m)
        if dj == 0:
            for f in of:
                st |= {(x, y + f) for x, y in ln}
        else:
            for f in of:
                st |= {(x + f, y) for x, y in ln}
    for x, y in st:
        if i(x, y) and G[x][y] == bg:
            G[x][y] = 3
    return G
