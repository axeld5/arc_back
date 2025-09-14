def p(grid, enumerator=enumerate, minimum=min):

    def find_positions(grid, target_value):
        return [(row_idx, col_idx) for row_idx, row in enumerator(grid) for col_idx, val in enumerator(row) if val == target_value]

    def min_coordinates(positions):
        min_row = minimum((row for row, _ in positions))
        min_col = minimum((col for _, col in positions))
        return (min_row, min_col)

    def move_values(origin_positions, target_offset, grid):
        for r, c in origin_positions:
            grid[r][c] = 0
        for r, c in origin_positions:
            new_r, new_c = (r + target_offset[0], c + target_offset[1])
            if 0 <= new_r < len(grid) and 0 <= new_c < len(grid[0]):
                grid[new_r][new_c] = 2
    positions_of_two = find_positions(grid, 2)
    positions_of_three = find_positions(grid, 3)
    origin_min_coords = min_coordinates(positions_of_two)
    destination_min_coords = min_coordinates(positions_of_three)
    target_offset = (destination_min_coords[0] - origin_min_coords[0] + 1, destination_min_coords[1] - origin_min_coords[1] + 1)
    move_values(positions_of_two, target_offset, grid)
    return grid
