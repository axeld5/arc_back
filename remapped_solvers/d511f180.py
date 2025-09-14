def p(grid):
    return [[element ^ 13 if element in (5, 8) else element for element in row] for row in grid]
