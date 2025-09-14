def find_snippet_positions(grid, target_value, grid_size):
    return [(row, col) for row in range(grid_size) for col in range(grid_size) if grid[row][col] == target_value]

def generate_possible_positions(positions):
    min_row = min((r for r, _ in positions))
    min_col = min((c for _, c in positions))
    possible_positions = []
    for x_offset in range(0, min_row + 1):
        for y_offset in range(0, min_col + 1):
            translated_positions = {(r - x_offset, c - y_offset) for r, c in positions}
            if all((0 <= r < 3 and 0 <= c < 3 for r, c in translated_positions)):
                possible_positions.append((x_offset, y_offset, translated_positions))
    return possible_positions

def p(grid, default_value=0, range_func=range):
    GRID_SIZE = 10
    extended_grid = [[default_value] * (GRID_SIZE + 1) for _ in range_func(GRID_SIZE + 1)]
    for r in range_func(GRID_SIZE):
        for c in range_func(GRID_SIZE):
            extended_grid[r + 1][c + 1] = grid[r][c]
    target_values = sorted({value for row in extended_grid for value in row if value != default_value})
    first_value, second_value = target_values
    position_map = {value: find_snippet_positions(extended_grid, value, GRID_SIZE + 1) for value in target_values}
    positions_first_value = generate_possible_positions(position_map[first_value])
    positions_second_value = generate_possible_positions(position_map[second_value])
    full_grid_set = {(r, c) for r in range_func(3) for c in range_func(3)}
    first_snippet = second_snippet = None
    for _, _, snippet_positions_first in positions_first_value:
        for _, _, snippet_positions_second in positions_second_value:
            if snippet_positions_first.isdisjoint(snippet_positions_second) and snippet_positions_first | snippet_positions_second == full_grid_set:
                first_snippet, second_snippet = (snippet_positions_first, snippet_positions_second)
                break
        if first_snippet is not None:
            break
    result = [[default_value] * 3 for _ in range_func(3)]
    for r, c in first_snippet:
        result[r][c] = first_value
    for r, c in second_snippet:
        result[r][c] = second_value
    return result
