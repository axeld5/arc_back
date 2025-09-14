def find_five_positions(grid, enumerate_func=enumerate):
    row_indices = [i for i, row in enumerate_func(grid) for j, value in enumerate_func(row) if value == 5]
    col_indices = [j for row in grid for j, value in enumerate_func(row) if value == 5]
    return (row_indices, col_indices)

def extract_subgrid(grid, top, bottom, left, right):
    return [row[left:right + 1] for row in grid[top:bottom + 1]]

def adjust_boundaries(top, left, bottom, right):
    adjusted_top = top - 1
    height = bottom - top + 3
    width = right - left + 1
    return (adjusted_top, height, left, width)

def p(grid, enumerate_func=enumerate):
    row_indices, col_indices = find_five_positions(grid, enumerate_func)
    top, bottom, left, right = (min(row_indices), max(row_indices), min(col_indices), max(col_indices))
    adjusted_top, height, left, width = adjust_boundaries(top, left, bottom, right)
    return extract_subgrid(grid, adjusted_top, adjusted_top + height - 1, left, left + width - 1)
