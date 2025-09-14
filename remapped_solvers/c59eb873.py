def p(grid):
    result = []
    for row in grid:
        doubled_row = []
        for element in row:
            doubled_row.extend([element, element])
        result.append(doubled_row)
        result.append(doubled_row)
    return result
