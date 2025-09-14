def p(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))
    base_values = [0, 0]
    marker_positions = []
    for row in range(num_rows):
        for col in range(num_cols):
            value = grid[row][col]
            if value not in (0, 8):
                base_values[(row + col) % 2] = value
                marker_positions.append((row, col))
    output_grid = [[0] * num_cols for _ in range(num_rows)]
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] == 8:
                output_grid[row][col] = 8
    to_visit = marker_positions[:]
    while to_visit:
        row, col = to_visit.pop()
        if row < 0 or row >= num_rows or col < 0 or (col >= num_cols) or (output_grid[row][col] != 0):
            continue
        output_grid[row][col] = base_values[(row + col) % 2]
        to_visit.extend([(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)])
    return output_grid
