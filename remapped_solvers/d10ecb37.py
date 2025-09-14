def extract_first_two_rows(grid):
    return [row[:2] for row in grid[:2]]

def p(grid):
    return extract_first_two_rows(grid)
