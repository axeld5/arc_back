def p(grid):
    for row in grid:
        unique_values = set(row) - {0}
        for value in unique_values:
            first_index = row.index(value)
            last_index = len(row) - row[::-1].index(value) - 1
            for i in range(first_index, last_index + 1):
                if row[i] == 0:
                    row[i] = value
    return grid
