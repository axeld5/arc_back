def find_non_empty_rows(grid):
    return [row_index for row_index, row in enumerate(grid) if any(row)]

def find_non_empty_columns(grid):
    return [col_index for col_index in range(len(grid[0])) if any((row[col_index] for row in grid))]

def extract_subgrid(grid, non_empty_rows, non_empty_columns):
    start_row, end_row = (non_empty_rows[0], non_empty_rows[-1] + 1)
    start_col, end_col = (non_empty_columns[0], non_empty_columns[-1] + 1)
    return [row[start_col:end_col][::-1] for row in grid[start_row:end_row]]

def p(grid):
    non_empty_rows = find_non_empty_rows(grid)
    non_empty_columns = find_non_empty_columns(grid)
    return extract_subgrid(grid, non_empty_rows, non_empty_columns)
