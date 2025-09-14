def p(grid):
    height, width = (len(grid), len(grid[0]))
    result = [[0, 0, 0] for _ in range(3)]
    for row in range(height):
        for col in range(width):
            if grid[row][col] == 5:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        neighbor_row = row + i
                        neighbor_col = col + j
                        if neighbor_row >= 0 and neighbor_col >= 0 and (neighbor_row < height) and (neighbor_col < width) and (grid[neighbor_row][neighbor_col] != 0):
                            result[1 + i][1 + j] = grid[neighbor_row][neighbor_col]
    return result
