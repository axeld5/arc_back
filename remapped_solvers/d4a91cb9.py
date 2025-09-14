def p(grid):

    def find_element(element):
        for row_idx, row in enumerate(grid):
            if element in row:
                return (row_idx, row.index(element))
    row_8, col_8 = find_element(8)
    row_2, col_2 = find_element(2)
    start_row = min(row_8, row_2)
    end_row = max(row_8, row_2)
    for row in range(start_row, end_row + 1):
        if row != row_8:
            grid[row][col_8] = 4
    start_col = min(col_8, col_2)
    end_col = max(col_8, col_2)
    for col in range(start_col, end_col + 1):
        if col != col_2:
            grid[row_2][col] = 4
    return grid
