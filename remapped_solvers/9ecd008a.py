def reverse_rows(grid):
    return [row[::-1] for row in grid]

def find_zero_positions(grid):
    zero_positions = []
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value == 0:
                zero_positions.append((row_index, col_index))
    return zero_positions

def find_zero_boundaries(zero_positions):
    row_indices, col_indices = zip(*zero_positions)
    min_row = min(row_indices)
    max_row = max(row_indices)
    min_col = min(col_indices)
    max_col = max(col_indices)
    return (min_row, max_row, min_col, max_col)

def extract_subgrid(grid, min_row, max_row, min_col, max_col):
    return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

def p(grid, E=enumerate):
    reversed_grid = reverse_rows(grid)
    zero_positions = find_zero_positions(grid)
    min_row, max_row, min_col, max_col = find_zero_boundaries(zero_positions)
    return extract_subgrid(reversed_grid, min_row, max_row, min_col, max_col)
