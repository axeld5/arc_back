def get_unique_corner(grid):
    corners_and_opposites = [[[grid[0][0], grid[0][1]], [grid[1][0], grid[1][1]]], [[grid[3][0], grid[3][1]], [grid[4][0], grid[4][1]]], [[grid[0][3], grid[0][4]], [grid[1][3], grid[1][4]]], [[grid[3][3], grid[3][4]], [grid[4][3], grid[4][4]]]]
    frequency_map = {}
    for corner in corners_and_opposites:
        corner_str = str(corner)
        if corner_str in frequency_map:
            frequency_map[corner_str] += 1
        else:
            frequency_map[corner_str] = 1
    for corner_str in frequency_map:
        if frequency_map[corner_str] == 1:
            return eval(corner_str)

def p(grid):
    return get_unique_corner(grid)
