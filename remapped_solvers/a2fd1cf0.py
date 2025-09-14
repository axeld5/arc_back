def p(grid):
    pos_2 = None
    pos_3 = None
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 2:
                pos_2 = (row, col)
            if grid[row][col] == 3:
                pos_3 = (row, col)
    row_2, col_2 = pos_2
    row_3, col_3 = pos_3
    step = 1 if col_3 > col_2 else -1
    for col in range(col_2 + step, col_3 + step, step):
        grid[row_2][col] = 8
    step = 1 if row_3 > row_2 else -1
    for row in range(row_2 + step, row_3, step):
        grid[row][col_3] = 8
    return grid
