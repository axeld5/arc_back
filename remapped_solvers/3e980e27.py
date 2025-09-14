from collections import deque, Counter

def p(grid):
    HEIGHT = 13
    BLACK, RED, GREEN = (0, 2, 3)
    result = [row[:] for row in grid]
    visited = [[False] * HEIGHT for _ in range(HEIGHT)]
    patterns = {}
    isolated_cells = {RED: [], GREEN: []}

    def get_neighbors(row, col):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    new_row, new_col = (row + dx, col + dy)
                    if 0 <= new_row < HEIGHT and 0 <= new_col < HEIGHT:
                        yield (new_row, new_col)
    for row in range(HEIGHT):
        for col in range(HEIGHT):
            if grid[row][col] == BLACK or visited[row][col]:
                continue
            queue = deque([(row, col)])
            visited[row][col] = True
            component = [(row, col)]
            while queue:
                curr_row, curr_col = queue.popleft()
                for next_row, next_col in get_neighbors(curr_row, curr_col):
                    if not visited[next_row][next_col] and grid[next_row][next_col] != BLACK:
                        visited[next_row][next_col] = True
                        queue.append((next_row, next_col))
                        component.append((next_row, next_col))
            if len(component) == 1:
                if grid[row][col] in (RED, GREEN):
                    isolated_cells[grid[row][col]].append((row, col))
                continue
            min_row = min((r for r, c in component))
            max_row = max((r for r, c in component))
            min_col = min((c for r, c in component))
            max_col = max((c for r, c in component))
            if max_row - min_row > 2 or max_col - min_col > 2:
                continue
            has_red = any((grid[r][c] == RED for r, c in component))
            has_green = any((grid[r][c] == GREEN for r, c in component))
            pattern_color = RED if has_red else GREEN if has_green else None
            if pattern_color is None:
                continue
            other_values = [grid[r][c] for r, c in component if grid[r][c] not in (RED, GREEN)]
            if not other_values:
                continue
            most_common_value = Counter(other_values).most_common(1)[0][0]
            relative_positions = {(r - min_row, c - min_col) for r, c in component}
            color_position = next(((r - min_row, c - min_col) for r, c in component if grid[r][c] == pattern_color))
            patterns[pattern_color] = (relative_positions, color_position, most_common_value)

    def apply_pattern(row, col, pattern_color, flip_horizontal):
        if pattern_color not in patterns:
            return
        relative_positions, color_pos, fill_value = patterns[pattern_color]
        color_row, color_col = color_pos
        if flip_horizontal:
            color_col = 2 - color_col
        base_row = row - color_row
        base_col = col - color_col
        for rel_row, rel_col in relative_positions:
            if flip_horizontal:
                rel_col = 2 - rel_col
            target_row = base_row + rel_row
            target_col = base_col + rel_col
            if 0 <= target_row < HEIGHT and 0 <= target_col < HEIGHT:
                if rel_row == color_row and rel_col == color_col:
                    result[target_row][target_col] = pattern_color
                else:
                    result[target_row][target_col] = fill_value
    for row, col in isolated_cells[RED]:
        apply_pattern(row, col, RED, flip_horizontal=True)
    for row, col in isolated_cells[GREEN]:
        apply_pattern(row, col, GREEN, flip_horizontal=False)
    return result
