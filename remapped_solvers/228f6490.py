def p(I, GRAY=5, BLACK=0):
    H, W = (len(I), len(I[0]))

    def neighbors4(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = (r + dr, c + dc)
            if 0 <= nr < H and 0 <= nc < W:
                yield (nr, nc)

    def component(start, allowed):
        from collections import deque
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in seen and allowed(I[nr][nc]):
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen
    seen = set()
    boxes = []
    for r in range(H):
        for c in range(W):
            if (r, c) in seen or I[r][c] != GRAY:
                continue
            comp = component((r, c), lambda v: v == GRAY)
            seen |= comp
            rs = [x for x, _ in comp]
            cs = [y for _, y in comp]
            r0, r1 = (min(rs), max(rs))
            c0, c1 = (min(cs), max(cs))
            interior = {(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)}
            boxes.append({'bbox': (r0, r1, c0, c1), 'interior': interior})
    silhouettes = []
    for b in boxes:
        S = {(r, c) for r, c in b['interior'] if I[r][c] == BLACK}
        silhouettes.append(S)
    all_box_cells = set().union(*[b['interior'] | {(r, c) for r in range(b['bbox'][0], b['bbox'][1] + 1) for c in range(b['bbox'][2], b['bbox'][3] + 1)} for b in boxes])
    visited = set()
    sprites = []
    for r in range(H):
        for c in range(W):
            if (r, c) in visited:
                continue
            v = I[r][c]
            if v in (GRAY, BLACK) or (r, c) in all_box_cells or v == 0:
                continue
            comp = component((r, c), lambda x: x == v)
            visited |= comp
            if comp.isdisjoint(all_box_cells):
                sprites.append((v, comp))
    total_color_counts = {}
    for r in range(H):
        for c in range(W):
            v = I[r][c]
            if v not in (GRAY, BLACK, 0):
                total_color_counts[v] = total_color_counts.get(v, 0) + 1
    from collections import defaultdict
    size_to_sprites = defaultdict(list)
    for color, cells in sprites:
        size = len(cells)
        if size == total_color_counts.get(color, 0):
            size_to_sprites[size].append((color, cells))
    O = [row[:] for row in I]
    for S_b in silhouettes:
        if not S_b:
            continue
        size = len(S_b)
        if size in size_to_sprites and size_to_sprites[size]:
            color, sprite_cells = size_to_sprites[size].pop(-1)
            for r, c in S_b:
                O[r][c] = color
            for r, c in sprite_cells:
                O[r][c] = 0
    return O
