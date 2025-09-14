from collections import deque

def p(grid):
    GRID_SIZE = 10
    EMPTY_VALUE = 0
    MARKED_VALUE = 2
    visited = set()
    groups_of_eight = []

    def find_group(start):
        queue = deque([start])
        visited_coordinates = {start}
        start_value = grid[start[0]][start[1]]
        while queue:
            row, col = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = (row + dr, col + dc)
                if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE and ((new_row, new_col) not in visited_coordinates) and (grid[new_row][new_col] == start_value):
                    visited_coordinates.add((new_row, new_col))
                    queue.append((new_row, new_col))
        return visited_coordinates
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if (row, col) in visited or grid[row][col] == EMPTY_VALUE:
                continue
            group = find_group((row, col))
            visited |= group
            if len(group) == 8:
                groups_of_eight.append(group)
    result_grid = [list(row) for row in grid]
    surrounding_coordinates = set()
    for group in groups_of_eight:
        for row, col in group:
            result_grid[row][col] = EMPTY_VALUE
        min_row = min((r for r, _ in group))
        min_col = min((c for _, c in group))
        surrounding_coordinates |= {(min_row + 1, min_col + 1), (min_row, min_col + 1), (min_row + 2, min_col + 1), (min_row + 1, min_col), (min_row + 1, min_col + 2)}
    for row, col in surrounding_coordinates:
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            result_grid[row][col] = MARKED_VALUE
    return result_grid
