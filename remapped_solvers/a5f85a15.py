def p(g):
    YELLOW = 4
    H, W = (len(g), len(g[0]))
    colors = {v for row in g for v in row if v not in (0, YELLOW)}
    if not colors:
        return [[0 for _ in range(W)] for _ in range(H)]
    K = colors.pop()
    out = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if g[r][c] == K:
                out[r][c] = K if c % 2 == 0 else YELLOW
    return out
