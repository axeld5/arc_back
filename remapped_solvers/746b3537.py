def p(grid):
    H, W = (len(grid), len(grid[0]))

    def row_is_uniform(r):
        return all((v == grid[r][0] for v in grid[r]))

    def col_is_uniform(c):
        first = grid[0][c]
        return all((grid[r][c] == first for r in range(H)))
    rows_uniform = all((row_is_uniform(r) for r in range(H)))
    cols_uniform = all((col_is_uniform(c) for c in range(W)))
    if rows_uniform and (not cols_uniform):
        seq = []
        prev = None
        for r in range(H):
            color = grid[r][0]
            if color != prev:
                seq.append(color)
                prev = color
        return [[c] for c in seq]
    if cols_uniform:
        seq = []
        prev = None
        for c in range(W):
            color = grid[0][c]
            if color != prev:
                seq.append(color)
                prev = color
        return [seq]
    uniform_rows = sum((row_is_uniform(r) for r in range(H)))
    uniform_cols = sum((col_is_uniform(c) for c in range(W)))
    if uniform_rows >= uniform_cols:
        seq, prev = ([], None)
        for r in range(H):
            color = grid[r][0]
            if color != prev:
                seq.append(color)
                prev = color
        return [[c] for c in seq]
    else:
        seq, prev = ([], None)
        for c in range(W):
            color = grid[0][c]
            if color != prev:
                seq.append(color)
                prev = color
        return [seq]
