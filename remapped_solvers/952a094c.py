def transpose(matrix):
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]

def find_first_and_last_non_empty_row(grid):
    non_empty_rows = [index for index, row in enumerate(grid) if any(row)]
    return (non_empty_rows[0], non_empty_rows[-1])

def swap_four_elements(grid, rc1, cc1, rc2, cc2):
    grid[rc1][cc1], grid[rc2][cc2] = (grid[rc2][cc2], grid[rc1][cc1])

def p(grid):
    first_non_empty_row, last_non_empty_row = find_first_and_last_non_empty_row(grid)
    first_non_empty_col, last_non_empty_col = find_first_and_last_non_empty_row(transpose(grid))
    swap_four_elements(grid, first_non_empty_row + 1, first_non_empty_col + 1, last_non_empty_row + 1, last_non_empty_col + 1)
    swap_four_elements(grid, first_non_empty_row + 1, last_non_empty_col - 1, last_non_empty_row + 1, first_non_empty_col - 1)
    swap_four_elements(grid, last_non_empty_row - 1, first_non_empty_col + 1, first_non_empty_row - 1, last_non_empty_col + 1)
    swap_four_elements(grid, last_non_empty_row - 1, last_non_empty_col - 1, first_non_empty_row - 1, first_non_empty_col - 1)
    return grid
