from collections import Counter

def p(I):
    H, W = (len(I), len(I[0]))
    all_colors = []
    for row in I:
        all_colors.extend(row)
    bg = Counter(all_colors).most_common(1)[0][0]

    def long_line(axis):
        limit = (W if axis == 0 else H) - 2
        idxs, cols = ([], [])
        for i in range(H if axis == 0 else W):
            line = I[i] if axis == 0 else [I[r][i] for r in range(H)]
            c, n = Counter(line).most_common(1)[0]
            if c != bg and n >= limit:
                idxs.append(i)
                cols.append(c)
        return (idxs, cols)
    rows, rowc = long_line(0)
    cols, colc = long_line(1)
    up, down = (min(rows), max(rows))
    left, right = (min(cols), max(cols))
    top_c = rowc[rows.index(up)]
    bottom_c = rowc[rows.index(down)]
    left_c = colc[cols.index(left)]
    right_c = colc[cols.index(right)]
    interior = []
    for r in range(up + 1, down):
        for c in range(left + 1, right):
            interior.append(I[r][c])
    ray_c = None
    for c in (left_c, right_c, top_c, bottom_c):
        if c in interior:
            ray_c = c
            break
    if ray_c is not None:
        if ray_c == left_c:
            dr, dc = (0, -1)
        elif ray_c == right_c:
            dr, dc = (0, 1)
        elif ray_c == top_c:
            dr, dc = (-1, 0)
        else:
            dr, dc = (1, 0)
        ray_pts = [(r, c) for r in range(up + 1, down) for c in range(left + 1, right) if I[r][c] == ray_c]
    h, w = (down - up + 1, right - left + 1)
    O = [[bg for _ in range(w)] for _ in range(h)]
    for r in range(h):
        O[r][0] = left_c
        O[r][w - 1] = right_c
    for c in range(w):
        O[0][c] = top_c
        O[h - 1][c] = bottom_c
    if ray_c is not None:
        for r, c in ray_pts:
            rr, cc = (r - up, c - left)
            while 0 < rr < h - 1 and 0 < cc < w - 1:
                O[rr][cc] = ray_c
                rr += dr
                cc += dc
    O[0][0] = I[up][left]
    O[0][w - 1] = I[up][right]
    O[h - 1][0] = I[down][left]
    O[h - 1][w - 1] = I[down][right]
    return O
