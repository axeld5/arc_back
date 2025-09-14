def p(G):
    H, W = (len(G), len(G[0]))
    cand = set((G[r][0] for r in range(H))) | set((G[r][W - 1] for r in range(H))) | set((G[0][c] for c in range(W))) | set((G[H - 1][c] for c in range(W)))
    cand.remove(0)
    best = None
    rows = []
    cols = []
    for k in cand:
        rs = [r for r in range(H) if G[r][0] == k and G[r][W - 1] == k]
        cs = [c for c in range(W) if G[0][c] == k and G[H - 1][c] == k]
        if (rs or cs) and (best is None or len(rs) + len(cs) > len(rows) + len(cols)):
            best = k
            rows, cols = (rs, cs)
    O = [[0] * W for _ in range(H)]
    if best is None:
        return O
    for r in rows:
        O[r] = [best] * W
    for c in cols:
        for r in range(H):
            O[r][c] = best
    return O
