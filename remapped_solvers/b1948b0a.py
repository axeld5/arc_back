def replace_six_with_two_in_row(row):
    replacement_map = {6: 2}
    return [replacement_map.get(element, element) for element in row]

def p(grid):
    return [replace_six_with_two_in_row(row) for row in grid]
