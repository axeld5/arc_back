from collections import Counter, deque
from typing import List, Tuple, Dict, Set
Grid = List[List[int]]

def p(I: Grid) -> Grid:
    H, W = (len(I), len(I[0]))

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W
    N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    flat = [v for row in I for v in row]
    bg1, bg2 = [c for c, _ in Counter(flat).most_common(2)]

    def largest_bg_component(col: int) -> Set[Tuple[int, int]]:
        seen = [[False] * W for _ in range(H)]
        best: set = set()
        for r in range(H):
            for c in range(W):
                if I[r][c] != col or seen[r][c]:
                    continue
                q = deque([(r, c)])
                seen[r][c] = True
                cur = {(r, c)}
                while q:
                    x, y = q.popleft()
                    for dx, dy in N4:
                        nx, ny = (x + dx, y + dy)
                        if inb(nx, ny) and (not seen[nx][ny]) and (I[nx][ny] == col):
                            seen[nx][ny] = True
                            cur.add((nx, ny))
                            q.append((nx, ny))
                if len(cur) > len(best):
                    best = cur
        return best

    def bbox(cells: Set[Tuple[int, int]]):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (min(rs), max(rs), min(cs), max(cs))

    def solve_for_assignment(donor_bg: int, donor_bbox, recip_bg: int, recip_bbox):
        r0d, r1d, c0d, c1d = donor_bbox
        r0r, r1r, c0r, c1r = recip_bbox
        seen = [[False] * W for _ in range(H)]
        rects: List[List[Tuple[int, int, int]]] = []
        for r in range(r0d, r1d + 1):
            for c in range(c0d, c1d + 1):
                if I[r][c] == donor_bg or seen[r][c]:
                    continue
                q = deque([(r, c)])
                seen[r][c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y, I[x][y]))
                    for dx, dy in N4:
                        nx, ny = (x + dx, y + dy)
                        if r0d <= nx <= r1d and c0d <= ny <= c1d and (not seen[nx][ny]) and (I[nx][ny] != donor_bg):
                            seen[nx][ny] = True
                            q.append((nx, ny))
                rects.append(comp)
        anchors_by_color: Dict[int, Set[Tuple[int, int]]] = {}
        for r in range(r0r, r1r + 1):
            for c in range(c0r, c1r + 1):
                if I[r][c] != recip_bg:
                    anchors_by_color.setdefault(I[r][c], set()).add((r, c))
        out_h, out_w = (r1r - r0r + 1, c1r - c0r + 1)
        O = [[recip_bg for _ in range(out_w)] for _ in range(out_h)]
        for col, poss in anchors_by_color.items():
            for r, c in poss:
                O[r - r0r][c - c0r] = col

        def transforms(rel: List[Tuple[int, int, int]]):
            Ht = 1 + max((r for r, _, _ in rel))
            Wt = 1 + max((c for _, c, _ in rel))

            def map_all(f):
                return [(f(r, c)[0], f(r, c)[1], v) for r, c, v in rel]
            return [map_all(lambda r, c: (r, c)), map_all(lambda r, c: (c, Ht - 1 - r)), map_all(lambda r, c: (Ht - 1 - r, Wt - 1 - c)), map_all(lambda r, c: (Wt - 1 - c, r)), map_all(lambda r, c: (r, Wt - 1 - c)), map_all(lambda r, c: (Ht - 1 - r, c)), map_all(lambda r, c: (c, r)), map_all(lambda r, c: (Wt - 1 - c, Ht - 1 - r))]

        def placements(comp):
            rs = [r for r, _, _ in comp]
            cs = [c for _, c, _ in comp]
            br, bc = (min(rs), min(cs))
            rel = [(r - br, c - bc, v) for r, c, v in comp]
            plist = []
            for rel2 in transforms(rel):
                rect_anchors = [(dr, dc, v) for dr, dc, v in rel2 if v in anchors_by_color]
                if not rect_anchors:
                    continue
                cand = None
                for dr, dc, v in rect_anchors:
                    opts = {(ar - dr, ac - dc) for ar, ac in anchors_by_color[v]}
                    cand = opts if cand is None else cand & opts
                    if not cand:
                        break
                if not cand:
                    continue
                for base in cand:
                    br0, bc0 = base
                    cells = []
                    ok = True
                    for dr, dc, v in rel2:
                        rr, cc = (br0 + dr, bc0 + dc)
                        if not (r0r <= rr <= r1r and c0r <= cc <= c1r):
                            ok = False
                            break
                        cells.append((rr, cc, v))
                    if not ok:
                        continue
                    score = sum((1 for rr, cc, v in cells if O[rr - r0r][cc - c0r] in (recip_bg, v)))
                    consumed = {(rr, cc) for rr, cc, v in cells if v in anchors_by_color and (rr, cc) in anchors_by_color[v]}
                    plist.append((score, base, cells, consumed))
            plist.sort(key=lambda x: (x[0], len(x[3]), len(x[2])), reverse=True)
            return plist
        rects.sort(key=lambda comp: (-len([1 for _, _, v in comp if v in anchors_by_color]), -len(comp)))
        used_anchors: set = set()
        painted_total = 0
        for comp in rects:
            opts = placements(comp)
            chosen = None
            for score, base, cells, consumed in opts:
                if any((p in used_anchors for p in consumed)):
                    continue
                chosen = (score, base, cells, consumed)
                break
            if not chosen:
                continue
            score, base, cells, consumed = chosen
            used_anchors.update(consumed)
            painted_total += score
            for rr, cc, v in cells:
                if O[rr - r0r][cc - c0r] in (recip_bg, v):
                    O[rr - r0r][cc - c0r] = v
        return (O, painted_total)
    comp1 = largest_bg_component(bg1)
    comp2 = largest_bg_component(bg2)
    r10, r11, c10, c11 = bbox(comp1)
    r20, r21, c20, c21 = bbox(comp2)
    A1, score1 = solve_for_assignment(bg1, (r10, r11, c10, c11), bg2, (r20, r21, c20, c21))
    A2, score2 = solve_for_assignment(bg2, (r20, r21, c20, c21), bg1, (r10, r11, c10, c11))
    return A1 if score1 >= score2 else A2
