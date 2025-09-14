def p(grid):
    from collections import Counter

    def find_boundaries(cells):
        min_row = min((row for row, _ in cells))
        max_row = max((row for row, _ in cells))
        min_col = min((col for _, col in cells))
        max_col = max((col for _, col in cells))
        return (min_row, max_row, min_col, max_col)

    def mark_boundary(grid, min_row, max_row, min_col, max_col):
        for col in range(min_col, max_col + 1):
            if grid[min_row][col] == 0:
                grid[min_row][col] = 2
            if grid[max_row][col] == 0:
                grid[max_row][col] = 2
        for row in range(min_row, max_row + 1):
            if grid[row][min_col] == 0:
                grid[row][min_col] = 2
            if grid[row][max_col] == 0:
                grid[row][max_col] = 2

    def mark_internal_line(grid, internal_cells, min_row, max_row, min_col, max_col):
        if len(internal_cells) >= 2:
            row_counter = Counter((row for row, _ in internal_cells))
            col_counter = Counter((col for _, col in internal_cells))
            most_common_row = row_counter.most_common(1)
            most_common_col = col_counter.most_common(1)
            if most_common_row and most_common_col:
                row_frequency = most_common_row[0][1]
                col_frequency = most_common_col[0][1]
                if row_frequency >= 2 and row_frequency >= col_frequency:
                    chosen_row = most_common_row[0][0]
                    for col in range(min_col + 1, max_col):
                        if grid[chosen_row][col] == 0:
                            grid[chosen_row][col] = 2
                elif col_frequency >= 2:
                    chosen_col = most_common_col[0][0]
                    for row in range(min_row + 1, max_row):
                        if grid[row][chosen_col] == 0:
                            grid[row][chosen_col] = 2
    height, width = (len(grid), len(grid[0]))
    ones_positions = [(row, col) for row in range(height) for col in range(width) if grid[row][col] == 1]
    output_grid = [row[:] for row in grid]
    min_row, max_row, min_col, max_col = find_boundaries(ones_positions)
    mark_boundary(output_grid, min_row, max_row, min_col, max_col)
    internal_positions = [(row, col) for row, col in ones_positions if min_row < row < max_row and min_col < col < max_col]
    mark_internal_line(output_grid, internal_positions, min_row, max_row, min_col, max_col)
    return output_grid
