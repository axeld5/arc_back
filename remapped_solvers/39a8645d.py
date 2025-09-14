from collections import defaultdict, deque

def p(grid, range_=range, max_=max):
    GRID_SIZE = 14
    EMPTY_CELL = 0

    def get_neighbors(row, col):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                new_row, new_col = (row + dr, col + dc)
                if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
                    yield (new_row, new_col)
    visited = [[False] * GRID_SIZE for _ in range_(GRID_SIZE)]
    pattern_count = defaultdict(int)
    for row in range_(GRID_SIZE):
        for col in range_(GRID_SIZE):
            value = grid[row][col]
            if value == EMPTY_CELL or visited[row][col]:
                continue
            queue = deque([(row, col)])
            visited[row][col] = True
            connected_cells = [(row, col)]
            while queue:
                current_x, current_y = queue.popleft()
                for neighbor_x, neighbor_y in get_neighbors(current_x, current_y):
                    if not visited[neighbor_x][neighbor_y] and grid[neighbor_x][neighbor_y] == value:
                        visited[neighbor_x][neighbor_y] = True
                        queue.append((neighbor_x, neighbor_y))
                        connected_cells.append((neighbor_x, neighbor_y))
            min_row = min((x for x, _ in connected_cells))
            max_row = max((x for x, _ in connected_cells))
            min_col = min((y for _, y in connected_cells))
            max_col = max((y for _, y in connected_cells))
            if max_row - min_row >= 3 or max_col - min_col >= 3:
                continue
            normalized_pattern = tuple(sorted(((x - min_row, y - min_col) for x, y in connected_cells)))
            pattern_count[value, normalized_pattern] += 1
    (most_common_value, most_common_pattern), _ = max_(pattern_count.items(), key=lambda item: item[1])
    result_grid = [[0] * 3 for _ in range_(3)]
    for dx, dy in most_common_pattern:
        result_grid[dx][dy] = most_common_value
    return result_grid
