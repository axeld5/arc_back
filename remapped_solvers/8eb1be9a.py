from typing import List
Grid = List[List[int]]

def p(grid: Grid) -> Grid:
    H, W = (len(grid), len(grid[0]))
    top = next((r for r in range(H) if any((cell for cell in grid[r]))))
    h = 0
    while top + h < H and any((grid[top + h][c] for c in range(W))):
        h += 1
    out: Grid = [[0] * W for _ in range(H)]
    for r in range(H):
        src = top + (r - top) % h
        out[r] = grid[src][:]
    return out
