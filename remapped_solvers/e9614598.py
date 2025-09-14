def calculate_center_coordinates(filled_positions):
    sum_rows, sum_cols = map(sum, zip(*filled_positions))
    num_positions = len(filled_positions)
    center_row, center_col = (sum_rows // num_positions, sum_cols // num_positions)
    return (center_row, center_col)

def mark_center(grid, center_position):
    row, col = center_position
    for delta_row, delta_col in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        grid[row + delta_row][col + delta_col] = 3

def p(grid, enumerator=enumerate):
    filled_positions = [(row_index, col_index) for row_index, row in enumerator(grid) for col_index, cell in enumerator(row) if cell]
    center = calculate_center_coordinates(filled_positions)
    mark_center(grid, center)
    return grid
