def p(grid):
    first_column = [grid[i][0] for i in range(3)]
    last_column = [grid[i][2] for i in range(3)]
    if first_column == last_column:
        return [[1]]
    else:
        return [[7]]
