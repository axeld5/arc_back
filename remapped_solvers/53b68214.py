def p(G):
    H, W = (len(G), len(G[0]))
    positions = [(r, c) for r in range(H) for c in range(W) if G[r][c]]
    if not positions:
        return [[0] * W for _ in range(W)]
    position_set = set(positions)
    color = G[positions[0][0]][positions[0][1]]
    best_score = -1
    best_t = -1
    best_s = -1
    for t in (1, 2, 3):
        for s in (0, 1, 2):
            score = sum(((r + t, c + s) in position_set for r, c in positions if r + t < H))
            if score > best_score or (score == best_score and t > best_t) or (score == best_score and t == best_t and (s < best_s)):
                best_score = score
                best_t = t
                best_s = s
    output = [[0] * W for _ in range(W)]
    for r, c in positions:
        k = 0
        while r + k * best_t < W:
            new_col = c + k * best_s
            if 0 <= new_col < W:
                output[r + k * best_t][new_col] = color
            k += 1
    return output
