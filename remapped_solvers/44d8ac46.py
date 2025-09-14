from collections import deque

def p(grid, K=range, H=12):

    def get_neighbors(row, col):
        if row > 0:
            yield (row - 1, col)
        if col > 0:
            yield (row, col - 1)
        if row + 1 < H:
            yield (row + 1, col)
        if col + 1 < H:
            yield (row, col + 1)

    def explore_region(start, value, visited):
        queue = deque([start])
        discovered_cells = {start}
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in visited and grid[nr][nc] == value:
                    visited.add((nr, nc))
                    discovered_cells.add((nr, nc))
                    queue.append((nr, nc))
        return discovered_cells

    def get_bounding_box(cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return (min(rows), max(rows), min(cols), max(cols))

    def is_square_shape(cells):
        min_row, max_row, min_col, max_col = get_bounding_box(cells)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        return height == width and height * width == len(cells)
    visited_cells = set()
    regions = []
    for r in K(H):
        for c in K(H):
            if (r, c) not in visited_cells and grid[r][c] != 0:
                regions.append(explore_region((r, c), grid[r][c], visited_cells))
    incorrect_squares = set()
    for region in regions:
        min_row, max_row, min_col, max_col = get_bounding_box(region)
        full_square = {(r, c) for r in K(min_row, max_row + 1) for c in K(min_col, max_col + 1)}
        difference_set = full_square - region
        if is_square_shape(difference_set):
            incorrect_squares |= difference_set
    for r, c in incorrect_squares:
        grid[r][c] = 2
    return grid
