def extract_box(grid, start_row, start_col):
    box = [[grid[start_row + row_offset][start_col + col_offset] for col_offset in range(3)] for row_offset in range(3)]
    return box

def find_non_empty_row_index(grid):
    for row_index, row in enumerate(grid):
        if sum(row) > 0:
            return row_index
    return 0

def find_non_empty_column_index(grid):
    number_of_rows = len(grid)
    for col_index in range(number_of_rows):
        if sum((grid[row][col_index] for row in range(number_of_rows))) > 0:
            return col_index
    return 0

def p(grid, indices_range=range(9)):
    non_empty_row_index = find_non_empty_row_index(grid) // 3 * 3
    non_empty_column_index = find_non_empty_column_index(grid) // 3 * 3
    return [[grid[non_empty_row_index + y % 3][non_empty_column_index + x % 3] * bool(grid[non_empty_row_index + y // 3][non_empty_column_index + x // 3]) for x in indices_range] for y in indices_range]
