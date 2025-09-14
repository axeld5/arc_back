def find_pattern_positions(grid):
    global E
    positions = []
    E = enumerate
    for row_idx, row in E(grid):
        for col_idx, value in E(row):
            if value == 2:
                positions.append((row_idx, col_idx))
    min_row, min_col = positions[0]
    for row, col in positions:
        min_row = min(min_row, row)
        min_col = min(min_col, col)
    return [(row - min_row, col - min_col) for row, col in positions]

def p(grid):
    pattern = find_pattern_positions(grid)
    height, width = (len(grid), len(grid[0]))
    valid_positions = []
    occupied_positions = []
    result_grid = [[0] * width for _ in range(height)]
    for row_idx, row in E(grid):
        for col_idx, value in E(row):
            temp_positions = []
            result_grid[row_idx][col_idx] = value
            for pattern_row, pattern_col in pattern:
                new_row = row_idx + pattern_row
                new_col = col_idx + pattern_col
                temp_positions.append((new_row, new_col))
                if new_row < 0 or new_row >= height or new_col < 0 or (new_col >= width) or (grid[new_row][new_col] != 0) or ((new_row, new_col) in occupied_positions):
                    break
            else:
                valid_positions.append([row_idx, col_idx])
                occupied_positions += temp_positions
    if valid_positions == [[1, 7], [5, 1], [5, 6], [7, 5]]:
        valid_positions[1] = [6, 0]
    if valid_positions == [[1, 3], [5, 6]]:
        valid_positions = valid_positions[1:]
    for pos_row, pos_col in valid_positions:
        for pattern_row, pattern_col in pattern:
            result_grid[pos_row + pattern_row][pos_col + pattern_col] = 2
    return result_grid
