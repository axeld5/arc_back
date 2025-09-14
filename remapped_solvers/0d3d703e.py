def map_digit(digit, digit_map=[0, 5, 6, 4, 3, 1, 2, 7, 9, 8]):
    return digit_map[digit]

def map_row(row, digit_map=[0, 5, 6, 4, 3, 1, 2, 7, 9, 8]):
    return [map_digit(digit, digit_map) for digit in row]

def solve(grid, digit_map=[0, 5, 6, 4, 3, 1, 2, 7, 9, 8]):
    return [map_row(row, digit_map) for row in grid]
p = solve
