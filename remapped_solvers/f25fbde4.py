def get_first_non_zero_row_index(grid):
    return [any(row) for row in grid].index(True)

def get_last_non_zero_row_index(grid):
    return len(grid) - 1 - [any(row) for row in grid][::-1].index(True)

def get_non_zero_column_indices(grid):
    transposed_grid = list(zip(*grid))
    non_zero_columns = [i for i, col in enumerate(transposed_grid) if any(col)]
    first_column_index = non_zero_columns[0]
    last_column_index = non_zero_columns[-1]
    return (first_column_index, last_column_index)

def extract_subgrid(grid, first_row, last_row, first_col, last_col):
    subgrid = []
    for row in grid[first_row:last_row + 1]:
        subgrid.append([cell for cell in row[first_col:last_col + 1] for _ in range(2)])
    return subgrid

def solve(grid):
    first_row = get_first_non_zero_row_index(grid)
    last_row = get_last_non_zero_row_index(grid)
    first_col, last_col = get_non_zero_column_indices(grid)
    solution_subgrid = []
    for row in extract_subgrid(grid, first_row, last_row, first_col, last_col):
        solution_subgrid.append(row)
        solution_subgrid.append(row)
    return solution_subgrid
p = solve
