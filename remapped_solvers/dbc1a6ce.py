def mark_sequence_as_eight(grid, start, end, constant, is_row):
    if is_row:
        for col in range(min(start, end), max(start, end) + 1):
            grid[constant][col] = 8
    else:
        for row in range(min(start, end), max(start, end) + 1):
            grid[row][constant] = 8

def process_number(grid, num, positions):
    for i, (row1, col1) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            row2, col2 = positions[j]
            if row1 == row2:
                mark_sequence_as_eight(grid, col1, col2, row1, is_row=True)
            elif col1 == col2:
                mark_sequence_as_eight(grid, row1, row2, col1, is_row=False)
        grid[row1][col1] = 1

def p(original_grid, A=range):
    grid_copy = [row[:] for row in original_grid]
    for number in A(1, 10):
        positions = [(row, col) for row in A(len(original_grid)) for col in A(len(original_grid[0])) if original_grid[row][col] == number]
        process_number(grid_copy, number, positions)
    return grid_copy
