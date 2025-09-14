def process_character(value, active_state):
    if value != 0:
        active_state = (not active_state) * value
    return active_state

def transform_row(row, active_state):
    for index, char_value in enumerate(row):
        active_state = process_character(char_value, active_state)
        if char_value == 0:
            row[index] = active_state
    return row

def p(grid, active_state=0):
    for row in grid:
        transform_row(row, active_state)
    return grid
