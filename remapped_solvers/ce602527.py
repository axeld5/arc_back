from collections import Counter

def p(I):
    H, W = (len(I), len(I[0]))
    flat = [v for row in I for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    by_color = {}
    for r in range(H):
        for c in range(W):
            v = I[r][c]
            if v == bg:
                continue
            by_color.setdefault(v, []).append((r, c))

    def touches_border(color):
        return any((r == 0 or r == H - 1 or c == 0 or (c == W - 1) for r, c in by_color[color]))
    non_bg_colors = list(by_color.keys())
    border_colors = [col for col in non_bg_colors if touches_border(col)]
    if border_colors:
        magcolor = max(border_colors, key=lambda col: len(by_color[col]))
    else:
        magcolor = max(non_bg_colors, key=lambda col: len(by_color[col]))
    S_mag = set(by_color[magcolor])

    def scale2(points):
        minr = min((r for r, _ in points))
        minc = min((c for _, c in points))
        rel = [(r - minr, c - minc) for r, c in points]
        scaled = set()
        for r, c in rel:
            scaled.add((2 * r, 2 * c))
            scaled.add((2 * r + 1, 2 * c))
            scaled.add((2 * r, 2 * c + 1))
            scaled.add((2 * r + 1, 2 * c + 1))
        h = max((r for r, _ in points)) - minr + 1
        w = max((c for _, c in points)) - minc + 1
        return (scaled, (minr, minc, h, w))

    def matches_sprite(points):
        scaled, (minr, minc, h, w) = scale2(points)
        scaled_list = list(scaled)
        for mx, my in S_mag:
            for sx, sy in scaled_list:
                tr, tc = (mx - sx, my - sy)
                inside = {(rx + tr, ry + tc) for rx, ry in scaled if 0 <= rx + tr < H and 0 <= ry + tc < W}
                if inside == S_mag:
                    return (True, (minr, minc, h, w))
        return (False, None)
    candidates = [col for col in non_bg_colors if col != magcolor]
    chosen_color, meta = (None, None)
    for col in candidates:
        ok, info = matches_sprite(by_color[col])
        if ok:
            chosen_color, meta = (col, info)
            break
    if chosen_color is None:
        chosen_color = min(candidates, key=lambda col: len(by_color[col]))
        pts = by_color[chosen_color]
        minr = min((r for r, _ in pts))
        maxr = max((r for r, _ in pts))
        minc = min((c for _, c in pts))
        maxc = max((c for _, c in pts))
        meta = (minr, minc, maxr - minr + 1, maxc - minc + 1)
    minr, minc, h, w = meta
    out = [[bg for _ in range(w)] for _ in range(h)]
    for r, c in by_color[chosen_color]:
        out[r - minr][c - minc] = chosen_color
    return out
