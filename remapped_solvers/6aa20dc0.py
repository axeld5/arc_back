from collections import Counter, deque

def p(g):
    H, W = (len(g), len(g[0]))
    b = Counter((c for r in g for c in r)).most_common(1)[0][0]
    kr = kc = None
    for r in range(H - 2):
        for c in range(W - 2):
            s = {g[r + i][c + j] for i in range(3) for j in range(3)}
            if len(s) == 4 and b in s:
                kr, kc = (r, c)
                break
        if kr is not None:
            break
    m = [[g[kr + i][kc + j] for j in range(3)] for i in range(3)]

    def rot(a):
        return [list(x) for x in zip(*a[::-1])]

    def hfl(a):
        return [r[::-1] for r in a]

    def vfl(a):
        return a[::-1]
    ori, seen = ([], set())
    a = m
    for _ in range(4):
        for x in (a, hfl(a), vfl(a), vfl(hfl(a))):
            k = tuple(map(tuple, x))
            if k not in seen:
                seen.add(k)
                ori.append(x)
        a = rot(a)
    key = (kr, kc, kr + 3, kc + 3)

    def in_key(r, c):
        return key[0] <= r < key[2] and key[1] <= c < key[3]
    seen = set()
    sq = []
    for r in range(H):
        for c in range(W):
            if (r, c) in seen or g[r][c] == b or in_key(r, c):
                continue
            col = g[r][c]
            q, comp = (deque([(r, c)]), [])
            seen.add((r, c))
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = (x + dx, y + dy)
                    if 0 <= nx < H and 0 <= ny < W and ((nx, ny) not in seen) and (g[nx][ny] == col):
                        seen.add((nx, ny))
                        q.append((nx, ny))
            rs, cs = ([x for x, _ in comp], [y for _, y in comp])
            r0, r1, c0, c1 = (min(rs), max(rs), min(cs), max(cs))
            h, w = (r1 - r0 + 1, c1 - c0 + 1)
            if h == w and all((g[i][j] == col for i in range(r0, r1 + 1) for j in range(c0, c1 + 1))):
                sq.append((r0, c0, h, col))
    out = [row[:] for row in g]
    pos = {(r, c): (r, c, mag, col) for r, c, mag, col in sq}
    used = set()

    def paste(top, left, mag, mat):
        for i in range(3):
            for j in range(3):
                col = mat[i][j]
                if col == b:
                    continue
                R, C = (top + i * mag, left + j * mag)
                for di in range(mag):
                    for dj in range(mag):
                        out[R + di][C + dj] = col
    for r0, c0, mag, col in sq:
        if (r0, c0) in used:
            continue
        for dr in (-2 * mag, 2 * mag):
            for dc in (-2 * mag, 2 * mag):
                p = (r0 + dr, c0 + dc)
                if p not in pos or p in used:
                    continue
                r1, c1, _, col2 = pos[p]
                top, left = (min(r0, r1), min(c0, c1))
                corners = {(top, left): 'TL', (top, left + 2 * mag): 'TR', (top + 2 * mag, left): 'BL', (top + 2 * mag, left + 2 * mag): 'BR'}
                mapc = {corners[r0, c0]: col, corners[r1, c1]: col2}
                for mat in ori:
                    diag = {'TL': mat[0][0], 'TR': mat[0][2], 'BL': mat[2][0], 'BR': mat[2][2]}
                    if all((diag[k] == v for k, v in mapc.items())):
                        paste(top, left, mag, mat)
                        used.update({(r0, c0), (r1, c1)})
                        break
                if (r0, c0) in used:
                    break
            if (r0, c0) in used:
                break
    return out
