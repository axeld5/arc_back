from collections import defaultdict

def find_target_structure(grid):
    HEIGHT = 20
    POSITION_DICT = defaultdict(list)
    for row in range(HEIGHT):
        for col in range(HEIGHT):
            value = grid[row][col]
            if value:
                POSITION_DICT[value].append((row, col))
    for value, positions in POSITION_DICT.items():
        if len(positions) != 10:
            continue
        row_to_cols_map = defaultdict(list)
        for row, col in positions:
            row_to_cols_map[row].append(col)
        if len(row_to_cols_map) != 4:
            continue
        sorted_rows = sorted(row_to_cols_map)
        if sorted_rows != list(range(sorted_rows[0], sorted_rows[0] + 4)):
            continue
        counts = [len(row_to_cols_map[r]) for r in sorted_rows]
        if counts != [1, 3, 4, 2]:
            continue
        col = row_to_cols_map[sorted_rows[0]][0]
        if set(row_to_cols_map[sorted_rows[1]]) != {col - 1, col, col + 1}:
            continue
        if set(row_to_cols_map[sorted_rows[2]]) != {col - 2, col - 1, col + 1, col + 2}:
            continue
        if set(row_to_cols_map[sorted_rows[3]]) != {col - 3, col + 3}:
            continue
        return (value, sorted_rows[0], col)
    return (None, None, None)

def apply_fall_effect(grid, value_to_replace, filler, col_positions, min_rows_map):
    HEIGHT = 20
    new_grid = [row[:] for row in grid]
    for col in col_positions:
        start_row = min_rows_map[col]
        if any((grid[row][col] == filler and row >= start_row for row in range(HEIGHT))):
            row = HEIGHT - 1
            while new_grid[row][col] != value_to_replace:
                new_grid[row][col] = filler
                row -= 1
    return new_grid

def p(grid, set_type=set):
    target_value, start_row, start_col = find_target_structure(grid)
    if target_value is None:
        return [row[:] for row in grid]
    all_values = set_type((value for row in grid for value in row))
    all_values.discard(0)
    all_values.discard(target_value)
    filler_value = list(all_values)[0]
    column_positions = set()
    min_row_per_column = {}
    for offset, cols in ((0, [start_col]), (1, [start_col - 1, start_col, start_col + 1]), (2, [start_col - 2, start_col - 1, start_col + 1, start_col + 2]), (3, [start_col - 3, start_col + 3])):
        row = start_row + offset
        for col in cols:
            if 0 <= row < 20 and 0 <= col < 20 and (grid[row][col] == target_value):
                column_positions.add(col)
                min_row_per_column[col] = min(min_row_per_column.get(col, row), row)
    return apply_fall_effect(grid, target_value, filler_value, column_positions, min_row_per_column)
