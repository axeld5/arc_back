def p(I):
    H, W = (len(I), len(I[0]))
    rmin, rmax, cmin, cmax = (H, -1, W, -1)
    for r in range(H):
        for c in range(W):
            if I[r][c] != 0:
                rmin = min(rmin, r)
                rmax = max(rmax, r)
                cmin = min(cmin, c)
                cmax = max(cmax, c)
    if rmax < 0:
        return [row[:] for row in I]
    brow, bcol = (rmin, cmin)
    tall, wide = (rmax - rmin + 1, cmax - cmin + 1)
    border_color = I[brow][bcol]
    inner_color = I[brow + 2][bcol + 2]
    rows_offsets = set()
    cols_offsets = set()
    if tall >= 5 and wide >= 5:
        top_r = brow + 1
        bot_r = brow + tall - 2
        for c in range(bcol + 2, bcol + wide - 2 - 0):
            if I[top_r][c] == inner_color:
                cols_offsets.add(c - (bcol + 2))
            if I[bot_r][c] == inner_color:
                cols_offsets.add(c - (bcol + 2))
        left_c = bcol + 1
        right_c = bcol + wide - 2
        for r in range(brow + 2, brow + tall - 2):
            if I[r][left_c] == inner_color:
                rows_offsets.add(r - (brow + 2))
            if I[r][right_c] == inner_color:
                rows_offsets.add(r - (brow + 2))
    O = [[0 for _ in range(W)] for _ in range(H)]
    for roff in rows_offsets:
        R = brow + 2 + roff
        if 0 <= R < H:
            for c in range(W):
                O[R][c] = inner_color
    for coff in cols_offsets:
        C = bcol + 2 + coff
        if 0 <= C < W:
            for r in range(H):
                O[r][C] = inner_color
    for r in range(brow, brow + tall):
        for c in range(bcol, bcol + wide):
            O[r][c] = border_color
    for r in range(brow + 2, brow + tall - 2):
        for c in range(bcol + 2, bcol + wide - 2):
            O[r][c] = inner_color
    for roff in rows_offsets:
        R = brow + 2 + roff
        for c in range(bcol + 2, bcol + wide - 2):
            O[R][c] = border_color
    for coff in cols_offsets:
        C = bcol + 2 + coff
        for r in range(brow + 2, brow + tall - 2):
            O[r][C] = border_color
    return O
