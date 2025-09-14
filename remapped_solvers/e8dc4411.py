def solve(grid):
    from collections import Counter
    height, width = (len(grid), len(grid[0]))
    Range = range

    def get_most_common_and_zero_positions():
        zero_positions = []
        most_common_elem = Counter((val for row in grid for val in row)).most_common(1)[0][0]
        special_position = None
        special_value = None
        for r in Range(height):
            for j in Range(width):
                value = grid[r][j]
                if value == 0:
                    zero_positions.append((r, j))
                elif value != most_common_elem and special_position is None:
                    special_position = (r, j)
                    special_value = value
        return (most_common_elem, zero_positions, special_position, special_value)
    most_common_elem, zero_positions, special_pos, special_value = get_most_common_and_zero_positions()
    if special_pos is None or not zero_positions:
        return [row[:] for row in grid]
    special_r, special_j = special_pos
    zero_x = [r for r, _ in zero_positions]
    zero_y = [j for _, j in zero_positions]
    min_x, max_x = (min(zero_x), max(zero_x))
    min_y, max_y = (min(zero_y), max(zero_y))
    center_x, center_y = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    place_in_top_half = special_r <= center_x
    place_in_left_half = special_j <= center_y
    if place_in_top_half and place_in_left_half:
        target_position_r, target_position_j = (max_x, max_y)
    elif place_in_top_half and (not place_in_left_half):
        target_position_r, target_position_j = (max_x, min_y)
    elif not place_in_top_half and place_in_left_half:
        target_position_r, target_position_j = (min_x, max_y)
    else:
        target_position_r, target_position_j = (min_x, min_y)
    special_3x3 = False
    if max_x - min_x + 1 == 3 and max_y - min_y + 1 == 3:
        if special_r == max_x and (special_j == min_y - 1 or special_j == max_y + 1):
            special_3x3 = True
            target_position_r, target_position_j = (max_x - 1, (min_y + max_y) // 2)
    offsets = [(r - target_position_r, j - target_position_j) for r, j in zero_positions]
    delta_r = 1 if special_r > target_position_r else -1
    delta_j = 1 if special_j > target_position_j else -1
    extent_r = max(offsets, key=lambda x: x[0])[0] - min(offsets, key=lambda x: x[0])[0] + 1
    extent_j = max(offsets, key=lambda x: x[1])[1] - min(offsets, key=lambda x: x[1])[1] + 1
    new_grid = [row[:] for row in grid]
    current_r, current_j = (special_r, special_j)
    if extent_r == 3 and extent_j == 3:
        if special_r == max_x and (special_j == min_y - 1 or special_j == max_y + 1):
            current_r += 1
    while True:
        filled_any = False
        for offset_r, offset_j in offsets:
            new_r, new_j = (current_r + offset_r, current_j + offset_j)
            if 0 <= new_r < height and 0 <= new_j < width:
                filled_any = True
                if new_grid[new_r][new_j] == most_common_elem:
                    new_grid[new_r][new_j] = special_value
        if not filled_any:
            break
        current_r += delta_r * (extent_r - special_3x3)
        current_j += delta_j * (extent_j - special_3x3)
    return new_grid
p = solve
