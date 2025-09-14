from collections import Counter
from copy import deepcopy

def p(I):
    H, W = (len(I), len(I[0]))
    assert H == 3 and W > 0

    def most_frequent_color(grid):
        return Counter((v for row in grid for v in row)).most_common(1)[0][0]

    def to_cols(grid):
        return [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]

    def from_cols(cols):
        return [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]

    def is_empty_col(col, BG):
        return all((v == BG for v in col))

    def detect_gray(grid, BG):
        flat = [v for row in grid for v in row]
        if 5 in flat and 5 != BG:
            return 5
        nonbg = [v for v in flat if v != BG]
        return Counter(nonbg).most_common()[-1][0] if nonbg else None

    def allowed_shifts(H):
        m = H - 1
        return list(range(-m, m + 1))

    def gray_overlap_score(cols_canvas, cols_block, prev_range, cur_range, shift, BG, GRAY):
        a_prev, b_prev = prev_range
        a_cur, _ = cur_range
        cL, cR = (b_prev, a_cur)
        rows_L_gray = {r for r in range(H) if cols_canvas[cL][r] == GRAY}
        rows_R_gray = {r for r in range(H) if cols_block[cR][r] == GRAY}
        if rows_L_gray and rows_R_gray:
            return (len({r for r in rows_R_gray if 0 <= r + shift < H and r + shift in rows_L_gray}), True)
        rows_L = {r for r in range(H) if cols_canvas[cL][r] != BG}
        rows_R = {r for r in range(H) if cols_block[cR][r] != BG}
        score = 0
        for r in rows_R:
            rr = r + shift
            if 0 <= rr < H and (rr in rows_L or rr - 1 in rows_L or rr + 1 in rows_L):
                score += 1
        return (score, False)
    BG = most_frequent_color(I)
    GRAY = detect_gray(I, BG)
    cols_in = to_cols(I)
    kept_cols, blocks = ([], [])
    in_block, start_idx = (False, None)
    for col in cols_in:
        if is_empty_col(col, BG):
            if in_block:
                blocks.append((start_idx, len(kept_cols) - 1))
                in_block, start_idx = (False, None)
        else:
            if not in_block:
                in_block, start_idx = (True, len(kept_cols))
            kept_cols.append(col)
    if in_block:
        blocks.append((start_idx, len(kept_cols) - 1))
    if not kept_cols:
        return deepcopy(I)
    if not blocks:
        blocks = [(0, len(kept_cols) - 1)]
    canvas_cols = [[BG] * H for _ in range(len(kept_cols))]
    a0, b0 = blocks[0]
    for c in range(a0, b0 + 1):
        for r in range(H):
            v = kept_cols[c][r]
            if v != BG:
                canvas_cols[c][r] = v
    pref_order = [0, -1, 1, -2, 2]
    prev_range = blocks[0]
    for bi in range(1, len(blocks)):
        cur_range = blocks[bi]
        best_shift, best_score, used_gray = (None, -1, None)
        gray_mode_seen = False
        candidates = []
        for d in allowed_shifts(H):
            ok = True
            for c in range(cur_range[0], cur_range[1] + 1):
                for r in range(H):
                    if kept_cols[c][r] != BG:
                        rr = r + d
                        if rr < 0 or rr >= H:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue
            sc, is_gray_mode = gray_overlap_score(canvas_cols, kept_cols, prev_range, cur_range, d, BG, GRAY)
            candidates.append((d, sc, is_gray_mode))
            if is_gray_mode:
                gray_mode_seen = True
        if not candidates:
            candidates = [(0, 0, False)]
        if gray_mode_seen:
            candidates = [t for t in candidates if t[2]]
        for d, sc, _ in candidates:
            if sc > best_score or (sc == best_score and (best_shift is None or pref_order.index(d) < pref_order.index(best_shift))):
                best_score, best_shift = (sc, d)
        shift = best_shift if best_shift is not None else 0
        for c in range(cur_range[0], cur_range[1] + 1):
            for r in range(H):
                v = kept_cols[c][r]
                if v != BG:
                    rr = r + shift
                    if 0 <= rr < H:
                        canvas_cols[c][rr] = v
        prev_range = cur_range
    if GRAY is not None:
        for a, b in blocks:
            cnt = Counter()
            for c in range(a, b + 1):
                for r in range(H):
                    v = canvas_cols[c][r]
                    if v != BG and v != GRAY:
                        cnt[v] += 1
            if not cnt:
                continue
            seg_color = cnt.most_common(1)[0][0]
            for c in range(a, b + 1):
                for r in range(H):
                    if canvas_cols[c][r] == GRAY:
                        canvas_cols[c][r] = seg_color
    return from_cols(canvas_cols)
