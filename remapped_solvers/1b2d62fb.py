def p(grid):
    result = []
    for row in grid:
        transformed_row = []
        for i in range(len(row) - 4):
            a = row[i]
            b = row[i + 4]
            value = 8 * (not (a or b))
            transformed_row.append(value)
        result.append(transformed_row)
    return result
