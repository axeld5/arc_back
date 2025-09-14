def decrement_seven_to_five(value):
    return value - 2 if value == 7 else value

def process_row(row):
    return [decrement_seven_to_five(element) for element in row]

def p(grid):
    return [process_row(row) for row in grid]
