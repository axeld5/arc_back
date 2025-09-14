def find_borders(grid):
    rows = range(10)
    columns = range(10)
    coordinates_of_eights = [(r, c) for r in rows for c in columns if grid[r][c] == 8]
    top_border = min((r for r, _ in coordinates_of_eights))
    bottom_border = max((r for r, _ in coordinates_of_eights))
    left_border = min((c for _, c in coordinates_of_eights))
    right_border = max((c for _, c in coordinates_of_eights))
    return (top_border, bottom_border, left_border, right_border)

def adjust_to_borders(grid, top, bottom, left, right):
    new_grid = [row[:] for row in grid]
    for r in range(10):
        for c in range(10):
            value = grid[r][c]
            if value != 0 and value != 8:
                bounded_row = min(max(r, top), bottom)
                bounded_col = min(max(c, left), right)
                new_grid[bounded_row][bounded_col] = value
    return new_grid

def p(I, m=min, M=max):
    top, bottom, left, right = find_borders(I)
    return adjust_to_borders(I, top, bottom, left, right)
