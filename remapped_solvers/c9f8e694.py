def process_row(row):
    for number in row:
        if number != 0 and number != 5:
            return [number if x == 5 else x for x in row]
    return row

def p(grid):
    for row in grid:
        processed_row = process_row(row)
        row[:] = processed_row
    return grid
