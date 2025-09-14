def p(grid):

    def find_eights(grid):
        positions = []
        for row_index in range(len(grid)):
            for col_index in range(len(grid[0])):
                if grid[row_index][col_index] == 8:
                    positions.append((row_index, col_index))
        return positions

    def find_rectangle_bounds(positions):
        if not positions:
            return (None, None, None, None)
        min_row = min((position[0] for position in positions))
        max_row = max((position[0] for position in positions))
        min_col = min((position[1] for position in positions))
        max_col = max((position[1] for position in positions))
        return (min_row, max_row, min_col, max_col)

    def mark_ones_in_rectangle(grid, min_row, max_row, min_col, max_col):
        updated_grid = [row[:] for row in grid]
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if grid[r][c] == 1:
                    updated_grid[r][c] = 3
        return updated_grid
    eights_positions = find_eights(grid)
    min_row, max_row, min_col, max_col = find_rectangle_bounds(eights_positions)
    if min_row is None:
        return grid
    return mark_ones_in_rectangle(grid, min_row, max_row, min_col, max_col)
