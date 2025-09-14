def p(grid):
    rows, cols = (len(grid), len(grid[0]))
    min_row, max_row, min_col, max_col = (rows, 0, cols, 0)
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != 1:
                if row < min_row:
                    min_row = row
                if row > max_row:
                    max_row = row
                if col < min_col:
                    min_col = col
                if col > max_col:
                    max_col = col
    return [[cell - (cell == 1) for cell in row[min_col:max_col + 1]] for row in grid[min_row:max_row + 1]]
