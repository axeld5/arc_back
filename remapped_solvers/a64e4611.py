GREEN = 3
MIN_WIDTH = 4

def p(I):
    G = [row[:] for row in I]
    solved = _solve_in_orientation(G)
    if solved is not None:
        return solved
    GT = _transpose(G)
    solvedT = _solve_in_orientation(GT)
    if solvedT is not None:
        return _transpose(solvedT)
    GF = _flip(G)
    solvedF = _solve_in_orientation(GF)
    if solvedF is not None:
        return _flip(solvedF)
    GFT = _transpose(_flip(G))
    solvedFT = _solve_in_orientation(GFT)
    if solvedFT is not None:
        return _flip(_transpose(solvedFT))
    return G

def _solve_in_orientation(G):
    H = len(G)
    W = len(G[0]) if H else 0
    if H == 0 or W == 0:
        return None
    forbidden = _forbidden_mask(G)
    artery = _find_artery_once(G, forbidden)
    if artery is None:
        return None
    with_artery = _draw_artery(G, artery, GREEN)
    with_veins = _draw_veins(with_artery, artery, forbidden, GREEN)
    return with_veins

def _find_artery_once(G, forbidden):
    H = len(G)
    W = len(G[0])
    r_base = H - 1
    possible_tops = [0, 5]
    j_min, j_max = (5, 12)
    for r_top in possible_tops:
        best_width = 0
        best_j = None
        for j in range(j_min, j_max + 1):
            width = _max_prefix_width(forbidden, r_top, r_base, j, W)
            if width > best_width:
                best_width = width
                best_j = j
        if best_width >= MIN_WIDTH:
            return (r_top, best_j, best_width)
    return None

def _max_prefix_width(forbidden, r_top, r_base, j_start, W):
    width = 0
    for c in range(j_start, W):
        if _col_segment_is_clear(forbidden, r_top, r_base, c):
            width += 1
        else:
            break
    return width

def _col_segment_is_clear(forbidden, r_top, r_base, c):
    for r in range(r_top, r_base + 1):
        if forbidden[r][c]:
            return False
    return True

def _draw_veins(G, artery, forbidden, color):
    out = [row[:] for row in G]
    H = len(out)
    W = len(out[0])
    r_top, j_left, width = artery
    r_base = H - 1
    left_end = j_left - 1
    right_start = j_left + width
    for r in range(r_top, r_base + 1):
        if left_end >= 0 and _row_segment_clear(forbidden, r, 0, left_end):
            for c in range(0, left_end + 1):
                out[r][c] = color
        if right_start <= W - 1 and _row_segment_clear(forbidden, r, right_start, W - 1):
            for c in range(right_start, W):
                out[r][c] = color
    return out

def _row_segment_clear(forbidden, r, c0, c1):
    for c in range(c0, c1 + 1):
        if forbidden[r][c]:
            return False
    return True

def _forbidden_mask(G):
    H = len(G)
    W = len(G[0])
    mask = [[False] * W for _ in range(H)]
    nz = [(r, c) for r in range(H) for c in range(W) if G[r][c] != 0]
    if not nz:
        return mask
    for r0, c0 in nz:
        for dr in (-1, 0, 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in (-1, 0, 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                mask[rr][cc] = True
    return mask

def _draw_artery(G, cand, color):
    r_top, j_left, width = cand
    H = len(G)
    r_base = H - 1
    out = [row[:] for row in G]
    for r in range(r_top, r_base + 1):
        for c in range(j_left, j_left + width):
            out[r][c] = color
    return out

def _transpose(G):
    if not G:
        return []
    H = len(G)
    W = len(G[0])
    T = [[0] * H for _ in range(W)]
    for r in range(H):
        for c in range(W):
            T[c][r] = G[r][c]
    return T

def _flip(G):
    return G[::-1]
