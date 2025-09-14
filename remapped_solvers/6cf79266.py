def p(grid):
    for row in range(18):
        row1, row2, row3 = grid[row:row + 3]
        for col in range(18):
            col_end = col + 3
            region_sum = sum(row1[col:col_end]) + sum(row2[col:col_end]) + sum(row3[col:col_end])
            if region_sum == 0:
                row1[col:col_end] = [1] * 3
                row2[col:col_end] = [1] * 3
                row3[col:col_end] = [1] * 3
    return grid
