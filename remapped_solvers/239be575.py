from collections import deque
from typing import List, Tuple, Set
BLACK, CYAN, RED = (0, 8, 2)
Coord = Tuple[int, int]
Grid = List[List[int]]

def find_red_boxes(grid: Grid) -> List[Set[Coord]]:
    h, w = (len(grid), len(grid[0]))
    boxes: List[Set[Coord]] = []
    used: Set[Coord] = set()
    for r in range(h - 1):
        for c in range(w - 1):
            if grid[r][c] == RED and grid[r][c + 1] == RED and (grid[r + 1][c] == RED) and (grid[r + 1][c + 1] == RED):
                cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                if cells & used:
                    continue
                boxes.append(cells)
                used |= cells
    if len(boxes) != 2:
        raise ValueError(f'Expected exactly 2 red boxes, found {len(boxes)}')
    return boxes

def p(grid: Grid) -> int:
    if not grid or not grid[0]:
        return [[BLACK]]
    h, w = (len(grid), len(grid[0]))
    R1, R2 = find_red_boxes(grid)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_walkable(r: int, c: int) -> bool:
        return grid[r][c] != BLACK
    q = deque(R1)
    seen: Set[Coord] = set(R1)
    while q:
        r, c = q.popleft()
        if (r, c) in R2:
            return [[CYAN]]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = (r + dr, c + dc)
            if in_bounds(nr, nc) and is_walkable(nr, nc) and ((nr, nc) not in seen):
                seen.add((nr, nc))
                q.append((nr, nc))
    return [[BLACK]]
