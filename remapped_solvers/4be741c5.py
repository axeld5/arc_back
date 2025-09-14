def remove_duplicates(lst):
    unique_elements = []
    for element in lst:
        if element not in unique_elements:
            unique_elements.append(element)
    return unique_elements

def get_unique_rows(grid):
    return [remove_duplicates(row) for row in grid]

def flatten(grid):
    return [element for row in grid for element in row]

def p(grid):
    unique_rows = get_unique_rows(grid)
    if all((unique_rows[0] == row for row in unique_rows)):
        return [unique_rows[0]]
    return [[element] for element in remove_duplicates(flatten(grid))]
