def p(grid):
    H, W = (len(grid), len(grid[0]))

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W
    flowers = []
    for s in (3, 4):
        for r0 in range(H - s + 1):
            for c0 in range(W - s + 1):
                border = []
                interior = []
                ok = True
                for r in range(r0, r0 + s):
                    for c in range(c0, c0 + s):
                        v = grid[r][c]
                        if r in (r0, r0 + s - 1) or c in (c0, c0 + s - 1):
                            border.append(v)
                        else:
                            interior.append(v)
                if not border or not interior:
                    continue
                B = border[0]
                if any((v != B for v in border)):
                    continue
                A = interior[0]
                if any((v != A for v in interior)):
                    continue
                if A == B:
                    continue
                length = s - 2
                row = r0 + 1
                col = c0 + 1
                flowers.append((row, col, length, A, B))
    out = [row[:] for row in grid]
    for row, col, length, A, B in flowers:
        for r in range(row - length - 1, row + 2 * length + 1):
            for c in range(col - length - 1, col + 2 * length + 1):
                if not inb(r, c):
                    continue
                if (r < row - 1 or r > row + length) and (c < col - 1 or c > col + length):
                    continue
                out[r][c] = B
        for r in range(row - 1, row + length + 1):
            for c in range(col - 1, col + length + 1):
                if inb(r, c):
                    out[r][c] = A
        for r in range(row, row + length):
            for c in range(col, col + length):
                if inb(r, c):
                    out[r][c] = B
    return out
