def p(I):

    def find_stripes(grid, is_row=True):
        lines = []
        L = len(grid) if is_row else len(grid[0])
        N = len(grid[0]) if is_row else len(grid)
        for idx in range(L):
            vals = [grid[idx][j] if is_row else grid[j][idx] for j in range(N) if (grid[idx][j] if is_row else grid[j][idx]) != 0 and (grid[idx][j] if is_row else grid[j][idx]) != 5]
            if len(vals) == 0:
                continue
            lines.append((idx, vals[0]))
        return lines
    has_zero_row = any((all((cell == 0 for cell in row)) for row in I))
    if has_zero_row:
        is_row = True
    else:
        is_row = False
    stripes = find_stripes(I, is_row=is_row)
    stripes.sort(key=lambda x: x[0])
    colors = [color for _, color in stripes]
    n = len(colors)
    out = [[colors[r] for _ in range(n)] for r in range(n)]
    if not is_row:
        out = [list(r) for r in zip(*out)]
    return out
