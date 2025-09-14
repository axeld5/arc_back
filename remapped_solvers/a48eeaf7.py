def p(grid, enumerate_function=enumerate, range_function=range):

    def find_positions_of_value(value):
        return [(row_idx, col_idx) for row_idx, row in enumerate_function(grid) for col_idx, cell in enumerate_function(row) if cell == value]

    def calculate_boundary_positions(top, bottom, left, right):
        return {(i, left) for i in range_function(top, bottom + 1)} | {(i, right) for i in range_function(top, bottom + 1)} | {(top, j) for j in range_function(left, right + 1)} | {(bottom, j) for j in range_function(left, right + 1)}

    def replace_positions_in_grid(old_positions, boundary_positions):
        new_positions = set()
        for old_position in old_positions:
            closest_boundary = min(boundary_positions, key=lambda boundary_position: (abs(old_position[0] - boundary_position[0]) + abs(old_position[1] - boundary_position[1]), boundary_position))
            new_positions.add(closest_boundary)
        for row_idx, col_idx in old_positions:
            grid[row_idx][col_idx] = 0
        for row_idx, col_idx in new_positions:
            grid[row_idx][col_idx] = 5
    positions_of_two = find_positions_of_value(2)
    top_limit = min((row_idx for row_idx, _ in positions_of_two)) - 1
    bottom_limit = max((row_idx for row_idx, _ in positions_of_two)) + 1
    left_limit = min((col_idx for _, col_idx in positions_of_two)) - 1
    right_limit = max((col_idx for _, col_idx in positions_of_two)) + 1
    boundary_positions = calculate_boundary_positions(top_limit, bottom_limit, left_limit, right_limit)
    positions_of_five = find_positions_of_value(5)
    replace_positions_in_grid(positions_of_five, boundary_positions)
    return grid
