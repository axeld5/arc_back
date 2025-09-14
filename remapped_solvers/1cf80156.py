def find_non_zero_coordinates(grid):
    non_zero_coordinates = [(row_index, col_index) for row_index, row in enumerate(grid) for col_index, value in enumerate(row) if value != 0]
    return non_zero_coordinates

def find_min_max_coordinates(coordinates):
    rows, cols = zip(*coordinates)
    min_row, max_row = (min(rows), max(rows))
    min_col, max_col = (min(cols), max(cols))
    return (min_row, max_row, min_col, max_col)

def extract_subgrid(grid, min_row, max_row, min_col, max_col):
    return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

def p(grid, A=enumerate, m=min, M=max):
    non_zero_coords = find_non_zero_coordinates(grid)
    min_row, max_row, min_col, max_col = find_min_max_coordinates(non_zero_coords)
    return extract_subgrid(grid, min_row, max_row, min_col, max_col)
