def get_bounding_box(indices):
    min_i = min((i for i, _ in indices))
    max_i = max((i for i, _ in indices))
    min_j = min((j for _, j in indices))
    max_j = max((j for _, j in indices))
    return (min_i, max_i, min_j, max_j)

def extract_subgrid(indices, grid):
    min_i, max_i, min_j, max_j = get_bounding_box(indices)
    return [row[min_j:max_j + 1] for row in grid[min_i:max_i + 1]]

def solve(grid):
    filled_cell_indices = [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == 1]
    original_grid = [list(row) for row in grid]
    reversed_row_grid = [row[::-1] for row in original_grid]
    reversed_column_grid = original_grid[::-1]
    subgrid_from_reversed_column = extract_subgrid(filled_cell_indices, reversed_column_grid)
    subgrid_from_reversed_row = extract_subgrid(filled_cell_indices, reversed_row_grid)
    if any((1 in row for row in subgrid_from_reversed_row)):
        result_grid = subgrid_from_reversed_column
    else:
        result_grid = subgrid_from_reversed_row
    return result_grid
p = solve
