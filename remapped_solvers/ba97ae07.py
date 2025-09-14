def fill_row(row):
    return [row[0]] * len(row) if row[0] else row

def transpose(grid):
    return [[grid[y][x] for y in range(len(grid))] for x in range(len(grid[0]))]

def fill_rows(grid):
    return [fill_row(row) for row in grid]

def p(grid):
    filled = fill_rows(grid)
    if filled == grid:
        return transpose(fill_rows(transpose(grid)))
    else:
        return filled
