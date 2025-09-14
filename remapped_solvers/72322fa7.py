def p(I):
    H, W = (len(I), len(I[0]))
    O = [r[:] for r in I]

    def draw(r, c, kind, rim, cen):
        O[r][c] = cen
        if kind == 'plus':
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                O[r + dr][c + dc] = rim
        elif kind == 'x':
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                O[r + dr][c + dc] = rim
        elif kind == 'h':
            O[r][c - 1] = O[r][c + 1] = rim
        else:
            O[r - 1][c] = O[r + 1][c] = rim
    C2, R2 = ({}, {})
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            cen = I[r][c]
            if cen == 0:
                continue
            u, d, l, rh = (I[r - 1][c], I[r + 1][c], I[r][c - 1], I[r][c + 1])
            nw, ne, sw, se = (I[r - 1][c - 1], I[r - 1][c + 1], I[r + 1][c - 1], I[r + 1][c + 1])
            if u and u == d == l == rh and (nw == ne == sw == se == 0):
                C2[cen] = ('plus', u)
                R2[u] = ('plus', cen)
            if nw and nw == ne == sw == se and (u == d == l == rh == 0):
                C2[cen] = ('x', nw)
                R2[nw] = ('x', cen)
            if l and l == rh and (u == d == 0):
                C2[cen] = ('h', l)
                R2[l] = ('h', cen)
            if u and u == d and (l == rh == 0):
                C2[cen] = ('v', u)
                R2[u] = ('v', cen)
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            col = I[r][c]
            if col and col in C2:
                kind, rim = C2[col]
                draw(r, c, kind, rim, col)
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if O[r][c] != 0:
                continue
            u, d, l, rh = (I[r - 1][c], I[r + 1][c], I[r][c - 1], I[r][c + 1])
            nw, ne, sw, se = (I[r - 1][c - 1], I[r - 1][c + 1], I[r + 1][c - 1], I[r + 1][c + 1])
            if u and u == d == l == rh and (nw == ne == sw == se == 0) and (u in R2) and (R2[u][0] == 'plus'):
                draw(r, c, 'plus', u, R2[u][1])
                continue
            if nw and nw == ne == sw == se and (u == d == l == rh == 0) and (nw in R2) and (R2[nw][0] == 'x'):
                draw(r, c, 'x', nw, R2[nw][1])
                continue
            if l and l == rh and (u == d == 0) and (l in R2) and (R2[l][0] == 'h'):
                draw(r, c, 'h', l, R2[l][1])
                continue
            if u and u == d and (l == rh == 0) and (u in R2) and (R2[u][0] == 'v'):
                draw(r, c, 'v', u, R2[u][1])
    return O
