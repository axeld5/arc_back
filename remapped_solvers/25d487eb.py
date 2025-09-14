def p(grid):
    result = [row[:] for row in grid]
    rows, cols = (len(grid), len(grid[0]))
    value_counts = {}
    for row in range(rows):
        for col in range(cols):
            if grid[row][col]:
                value = grid[row][col]
                value_counts[value] = value_counts.get(value, 0) + 1
    unique_row, unique_col, unique_value = next(((row, col, grid[row][col]) for row in range(rows) for col in range(cols) if grid[row][col] and value_counts[grid[row][col]] == 1))
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        next_row, next_col = (unique_row + dr, unique_col + dc)
        if next_row < 0 or next_row >= rows or next_col < 0 or (next_col >= cols) or (grid[next_row][next_col] == 0):
            step = 1
            while 0 <= unique_row - step * dr < rows and 0 <= unique_col - step * dc < cols:
                target_row = unique_row - step * dr
                target_col = unique_col - step * dc
                if grid[target_row][target_col] == 0:
                    result[target_row][target_col] = unique_value
                step += 1
    return result
