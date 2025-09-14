def find_positions(grid, target_value, enumerate_func=enumerate):
    return [(row_idx, col_idx) for row_idx, row in enumerate_func(grid) for col_idx, value in enumerate_func(row) if value == target_value]

def determine_shift_direction(threes_positions):
    if len({row for row, _ in threes_positions}) == 1:
        return (0, 2)
    else:
        return (2, 0)

def move_twos(grid, twos_positions, shift_direction):
    updated_grid = [row[:] for row in grid]
    for row, col in twos_positions:
        updated_grid[row][col] = 0
    row_shift, col_shift = shift_direction
    for row, col in twos_positions:
        updated_grid[row + row_shift][col + col_shift] = 2
    return updated_grid

def p(grid, enumerate_func=enumerate):
    threes_positions = find_positions(grid, 3, enumerate_func)
    twos_positions = find_positions(grid, 2, enumerate_func)
    shift_direction = determine_shift_direction(threes_positions)
    return move_twos(grid, twos_positions, shift_direction)
