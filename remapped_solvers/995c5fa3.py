def calculate_cell_value(val1, val2, val3):
    return (45 - val3 - 2 * val2 - 4 * val1) // 5

def compute_new_grid(third_row, second_row):
    new_grid = []
    for index in range(0, 15, 5):
        new_value = calculate_cell_value(second_row[index + 1], third_row[index + 1], third_row[index])
        new_grid.append([new_value] * 3)
    return new_grid

def p(j):
    third_row = j[2]
    second_row = j[1]
    return compute_new_grid(third_row, second_row)
