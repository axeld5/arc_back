def modify_middle_element_if_condition_met(row):
    for index in range(len(row) - 2):
        if row[index] & row[index + 2]:
            row[index + 1] = 2
    return row

def p(grid):
    for row in grid:
        modify_middle_element_if_condition_met(row)
    return grid
