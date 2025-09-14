def p(grid):
    for row_idx, row in enumerate(grid):
        index = 0
        pattern = []
        has_positive = False
        for col_idx, value in enumerate(row):
            if value > 0:
                pattern = [value, 5] * 20
                has_positive = True
            if has_positive:
                grid[row_idx][col_idx] = pattern[index]
                index += 1
    return grid
