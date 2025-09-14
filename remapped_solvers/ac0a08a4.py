def count_positive_integers(grid):
    return sum((cell > 0 for row in grid for cell in row))

def repeat_row_elements(row, count):
    return sum(([element] * count for element in row), [])

def transform_grid(grid):
    positive_count = count_positive_integers(grid)
    return [repeat_row_elements(row, positive_count) for row in grid for _ in range(positive_count)]

def p(grid):
    return transform_grid(grid)
