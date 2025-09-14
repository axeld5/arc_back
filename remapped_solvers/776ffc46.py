from collections import deque

def p(grid, size=20):

    def neighbors(row, col):
        if row > 0:
            yield (row - 1, col)
        if col > 0:
            yield (row, col - 1)
        if row + 1 < size:
            yield (row + 1, col)
        if col + 1 < size:
            yield (row, col + 1)

    def explore_region(start_row, start_col, region_color):
        queue = deque([(start_row, start_col)])
        region_positions = {(start_row, start_col)}
        visited[start_row][start_col] = True
        while queue:
            current_row, current_col = queue.popleft()
            for neighbor_row, neighbor_col in neighbors(current_row, current_col):
                is_not_visited = not visited[neighbor_row][neighbor_col]
                is_same_color = grid[neighbor_row][neighbor_col] == region_color
                if is_not_visited and is_same_color:
                    visited[neighbor_row][neighbor_col] = True
                    region_positions.add((neighbor_row, neighbor_col))
                    queue.append((neighbor_row, neighbor_col))
        return region_positions

    def bounding_box(region):
        min_row = min((row for row, _ in region))
        min_col = min((col for _, col in region))
        max_row = max((row for row, _ in region))
        max_col = max((col for _, col in region))
        return (min_row, min_col, max_row, max_col)

    def create_border(bounds):
        upper, left, lower, right = bounds
        border = set()
        border.update(((r, left) for r in range(upper, lower + 1)))
        border.update(((r, right) for r in range(upper, lower + 1)))
        border.update(((upper, c) for c in range(left, right + 1)))
        border.update(((lower, c) for c in range(left, right + 1)))
        return border
    visited = [[False] * size for _ in range(size)]
    regions = []
    for row in range(size):
        for col in range(size):
            if not visited[row][col] and grid[row][col] != 0:
                color = grid[row][col]
                region_positions = explore_region(row, col, color)
                regions.append((color, region_positions))
    pattern_to_match = None
    for color, region in regions:
        if color != 5:
            continue
        bounds = bounding_box(region)
        if region == create_border(bounds):
            upper, left, lower, right = bounds
            pattern_to_match = (upper + 1, left + 1, lower - 1, right - 1)
            break
    top, left, bottom, right = pattern_to_match
    filled_region_boundary = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1) if grid[r][c] != 0}
    base_row = min((r for r, _ in filled_region_boundary))
    base_col = min((c for _, c in filled_region_boundary))
    normalized_block = {(r - base_row, c - base_col) for r, c in filled_region_boundary}
    block_origin = next(iter(filled_region_boundary))
    block_color = grid[block_origin[0]][block_origin[1]]
    matching_extended_region = set()
    for _, region in regions:
        normalized_region_start_row = min((r for r, _ in region))
        normalized_region_start_col = min((c for _, c in region))
        normalized_region = {(r - normalized_region_start_row, c - normalized_region_start_col) for r, c in region}
        if normalized_region == normalized_block:
            matching_extended_region.update(region)
    modified_grid = [row[:] for row in grid]
    for r, c in matching_extended_region:
        modified_grid[r][c] = block_color
    return modified_grid
