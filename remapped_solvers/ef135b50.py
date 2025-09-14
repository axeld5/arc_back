def p(grid, A=range):
    height = len(grid)
    for row in A(1, height - 1):
        state = start_col = 0
        for col in A(height):
            value = grid[row][col]
            if state < 1 and value > 1:
                state = 1
            if state == 1 and value < 1:
                state = 2
                start_col = col if start_col == 0 else start_col
            if state > 1 and value > 1:
                for fill_col in A(start_col, col):
                    grid[row][fill_col] = 9
                state = 1
                start_col = 0
    return grid
