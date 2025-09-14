def p(g):
    H, W = (len(g), len(g[0]))
    b = g[-1][-1]
    palette_set = {v for row in g for v in row if v != b}
    K = len(palette_set)
    order = []
    for c in range(W):
        v = g[0][c]
        if v == b:
            break
        if v not in order:
            order.append(v)
            if len(order) == K:
                break
    if len(order) < K:
        for v in palette_set:
            if v not in order:
                order.append(v)
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            out[r][c] = order[(r % 2 + c + 1) % K]
    return out
