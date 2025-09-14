BLACK, RED = (1, 2)
SCALE_FACTORS = [3, 2, 1]

def calculate_connected_black_cells(grid, start):
    visited = {start}
    queue = [start]
    while queue:
        row, col = queue.pop(0)
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row, new_col = (row + d_row, col + d_col)
            if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
                if grid[new_row][new_col] and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
    return visited

def parse_grid(grid):
    height, width = (len(grid), len(grid[0]))
    start_row, start_col = next(((i, j) for i in range(height) for j in range(width) if grid[i][j] == BLACK))
    connected_cells = calculate_connected_black_cells(grid, (start_row, start_col))
    min_row = min((row for row, _ in connected_cells))
    min_col = min((col for _, col in connected_cells))
    max_row = max((row for row, _ in connected_cells))
    max_col = max((col for _, col in connected_cells))
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j]:
                connected_cells.add((i, j))
    black_cells, red_cells = (set(), set())
    for i, j in connected_cells:
        if grid[i][j] == BLACK:
            black_cells.add((i - min_row, j - min_col))
        else:
            red_cells.add((i - min_row, j - min_col))
    return (max_row - min_row + 1, max_col - min_col + 1, black_cells, red_cells)

def consistent_black_red_placements(grid, height, width, black_cells, red_cells):
    consistent_red = set()
    for scale in SCALE_FACTORS:
        scaled_height, scaled_width = (height * scale, width * scale)
        for top in range(len(grid) - scaled_height + 1):
            for left in range(len(grid[0]) - scaled_width + 1):
                scaled_red_cells = {(top + dr * scale + r, left + dc * scale + c) for dr, dc in red_cells for r in range(scale) for c in range(scale)}
                scaled_black_cells = {(top + dr * scale + r, left + dc * scale + c) for dr, dc in black_cells for r in range(scale) for c in range(scale)}
                if all((grid[i][j] == RED for i, j in scaled_red_cells)) and all((grid[i][j] == BLACK for i, j in scaled_black_cells)):
                    if all(((i, j) in scaled_red_cells | scaled_black_cells or not grid[i][j] for i in range(top, top + scaled_height) for j in range(left, left + scaled_width))):
                        consistent_red |= scaled_red_cells
    return consistent_red

def find_solutions(grid, height, width, black_cells, red_cells, consistent_placement):
    for scale in SCALE_FACTORS:
        scaled_height, scaled_width = (height * scale, width * scale)
        for top in range(len(grid) - scaled_height + 1):
            for left in range(len(grid[0]) - scaled_width + 1):
                if any((grid[i][j] == BLACK for i in range(top, top + scaled_height) for j in range(left, left + scaled_width))):
                    continue
                scaled_red_cells = {(top + dr * scale + r, left + dc * scale + c) for dr, dc in red_cells for r in range(scale) for c in range(scale)}
                if any((grid[i][j] != RED or (i, j) in consistent_placement for i, j in scaled_red_cells)):
                    continue
                if any((grid[top + dr * scale + r][left + dc * scale + c] == RED for dr, dc in black_cells for r in range(scale) for c in range(scale))):
                    continue
                if any((grid[i][j] == RED and (i, j) not in scaled_red_cells for i in range(top, top + scaled_height) for j in range(left, left + scaled_width))):
                    continue
                yield (top, left, scale, scaled_red_cells)

def p(grid):
    grid = [row[:] for row in grid]
    for i in range(len(grid)):
        grid[i] = [0, 0, 0] + grid[i]
    height, width, black_cells, red_cells = parse_grid(grid)
    consistent_placement = consistent_black_red_placements(grid, height, width, black_cells, red_cells)
    for top, left, scale, scaled_red_cells in find_solutions(grid, height, width, black_cells, red_cells, consistent_placement):
        consistent_placement |= scaled_red_cells
        for dr, dc in black_cells:
            for r in range(scale):
                for c in range(scale):
                    grid[top + dr * scale + r][left + dc * scale + c] = BLACK
    for i in range(len(grid)):
        grid[i] = grid[i][3:]
    return grid
