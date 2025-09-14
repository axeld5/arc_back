def p(grid):
    grid_size = 21
    flat_values = [value for row in grid for value in row]
    least_frequent = min(set(flat_values), key=flat_values.count)
    target_positions = {(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == least_frequent}
    x_coords, y_coords = zip(*target_positions)
    min_x, min_y = (min(x_coords), min(y_coords))
    max_x, max_y = (max(x_coords), max(y_coords))
    left, top = (min_x + 1, min_y + 1)
    right, bottom = (max_x - 1, max_y - 1)
    border_positions = set()
    for i in range(left, right + 1):
        border_positions.add((i, top))
        border_positions.add((i, bottom))
    for j in range(top, bottom + 1):
        border_positions.add((left, j))
        border_positions.add((right, j))
    min_border_x = min((i for i, j in border_positions))
    min_border_y = min((j for i, j in border_positions))
    normalized_border = {(i - min_border_x, j - min_border_y) for i, j in border_positions}
    pattern_width = max((i for i, j in normalized_border)) + 1
    pattern_height = max((j for i, j in normalized_border)) + 1
    valid_placements = []
    for start_x in range(grid_size - pattern_width + 1):
        for start_y in range(grid_size - pattern_height + 1):
            if all((grid[start_x + i][start_y + j] == 0 for i, j in normalized_border)):
                valid_placements.append((start_x, start_y))
    target_relative = {(i - min_x, j - min_y) for i, j in target_positions}
    adjusted_target = {(i - 1, j - 1) for i, j in target_relative}
    placement_positions = set()
    for start_x, start_y in valid_placements:
        for rel_i, rel_j in adjusted_target:
            abs_i, abs_j = (start_x + rel_i, start_y + rel_j)
            if 0 <= abs_i < grid_size and 0 <= abs_j < grid_size:
                placement_positions.add((abs_i, abs_j))
    for i, j in placement_positions:
        grid[i][j] = least_frequent
    return grid
