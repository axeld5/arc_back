def toggle_element(value, anchor):
    return value and anchor

def modify_grid(grid):
    anchor = grid[6][0]
    modified_grid = [[toggle_element(element, anchor) for element in row] for row in grid]
    modified_grid[6][0] = 0
    return modified_grid

def p(j):
    return modify_grid(j)
