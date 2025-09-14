def p(j):
    grid_size = 10
    toggle = 0
    for row_idx, row in enumerate(j):
        for col_idx, value in enumerate(row):
            if value % 5:
                for col in range(col_idx, grid_size, 2):
                    for r in range(row_idx + 1):
                        j[r][col] = value
                for col in range(col_idx + 1, grid_size, 2):
                    j[toggle * (grid_size - 1)][col] = 5
                    toggle ^= 1
                return j
