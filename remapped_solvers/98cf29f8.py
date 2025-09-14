from collections import Counter

def p(grid, E=enumerate):
    height, width = (len(grid), len(grid[0]))
    count_values = Counter((value for row in grid for value in row))
    dominant_value = max(count_values, key=count_values.get)
    non_dominant_positions = {}
    for i, row in E(grid):
        for j, value in E(row):
            if value != dominant_value:
                non_dominant_positions.setdefault(value, []).append((i, j))
    position_groups = list(non_dominant_positions.values())

    def calculate_bounding_box(coords):
        rows, cols = zip(*coords)
        return (min(rows), min(cols), max(rows), max(cols))

    def calculate_area(bbox):
        top, left, bottom, right = bbox
        return (bottom - top + 1) * (right - left + 1)

    def is_full_rectangle(coords):
        return len(coords) == calculate_area(calculate_bounding_box(coords))
    focal_group, rectangle_group = (position_groups[0], position_groups[1]) if is_full_rectangle(position_groups[0]) else (position_groups[1], position_groups[0])
    common_value = next((grid[i][j] for i, j in rectangle_group))
    diagonal_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    inner_boundary_positions = []
    for i, j in rectangle_group:
        is_boundary = True
        for di, dj in diagonal_directions:
            ni, nj = (i + di, j + dj)
            if not (0 <= ni < height and 0 <= nj < width and (grid[ni][nj] == common_value)):
                is_boundary = False
                break
        if is_boundary:
            inner_boundary_positions.append((i, j))
    top, left, bottom, right = calculate_bounding_box(inner_boundary_positions)
    top, left, bottom, right = (top - 1, left - 1, bottom + 1, right + 1)
    bounding_box_positions = [(i, j) for i in range(top, bottom + 1) for j in range(left, right + 1)]
    for i, j in rectangle_group:
        grid[i][j] = dominant_value
    focal_top, focal_left, focal_bottom, focal_right = calculate_bounding_box(focal_group)
    exp_top, exp_left, exp_bottom, exp_right = (top, left, bottom, right)
    if not (exp_left <= focal_right and focal_left <= exp_right):
        row_offset = 0
        col_offset = focal_left - exp_right - 1 if exp_right < focal_left else focal_right - exp_left + 1
    else:
        col_offset = 0
        row_offset = focal_top - exp_bottom - 1 if exp_bottom < focal_top else focal_bottom - exp_top + 1
    shifted_positions = [(i + row_offset, j + col_offset) for i, j in bounding_box_positions]
    for i, j in shifted_positions:
        if 0 <= i < height and 0 <= j < width:
            grid[i][j] = common_value
    return grid
