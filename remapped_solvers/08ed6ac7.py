def replace_fives_with_unique_numbers(grid):
    column_replacements = {}
    for row in grid:
        for column_index, value in enumerate(row):
            if value == 5:
                row[column_index] = column_replacements.setdefault(column_index, len(column_replacements) + 1)
    return grid

def p(j):
    return replace_fives_with_unique_numbers(j)
