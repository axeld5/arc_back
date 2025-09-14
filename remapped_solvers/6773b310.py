def p(grid):
    result = [[0] * 3 for _ in range(3)]
    for block_row in range(3):
        for block_col in range(3):
            count = 0
            for row in range(3):
                for col in range(3):
                    if grid[block_row * 4 + row][block_col * 4 + col] == 6:
                        count += 1
            result[block_row][block_col] = 1 if count >= 2 else 0
    return result
