def p(grid):
    height, width = (len(grid), len(grid[0]))
    value_counts = {}
    for row in range(height):
        for col in range(width):
            if grid[row][col]:
                value_counts[grid[row][col]] = value_counts.get(grid[row][col], 0) + 1
    most_frequent = max(value_counts, key=value_counts.get)
    least_frequent = min(value_counts, key=value_counts.get)
    max_area = 0
    best_rect = None
    for top in range(height - 2):
        for left in range(width - 2):
            for bottom in range(top + 2, height):
                for right in range(left + 2, width):
                    top_border = all((grid[top][col] == most_frequent for col in range(left, right + 1)))
                    bottom_border = all((grid[bottom][col] == most_frequent for col in range(left, right + 1)))
                    left_border = all((grid[row][left] == most_frequent for row in range(top, bottom + 1)))
                    right_border = all((grid[row][right] == most_frequent for row in range(top, bottom + 1)))
                    if top_border and bottom_border and left_border and right_border:
                        area = (bottom - top + 1) * (right - left + 1)
                        if area > max_area:
                            max_area = area
                            best_rect = (top, left, bottom, right)
    top, left, bottom, right = best_rect
    result = []
    for row in range(bottom - top + 1):
        result_row = []
        for col in range(right - left + 1):
            original_value = grid[top + row][left + col]
            if original_value == most_frequent:
                result_row.append(least_frequent)
            else:
                result_row.append(original_value)
        result.append(result_row)
    return result
