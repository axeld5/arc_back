from collections import Counter

def find_corners_with_value(grid, value=5):
    return [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == value]

def get_min_max_coordinates(positions):
    rows, cols = zip(*positions)
    return (min(rows), max(rows), min(cols), max(cols))

def extract_inner_section(grid, top, bottom, left, right):
    return [row[left:right + 1] for row in grid[top:bottom + 1]]

def find_least_common_value(elements):
    return min(Counter(elements), key=Counter(elements).get)

def paint_border_with_value(grid, top, bottom, left, right, value):
    border_indices = set()
    for row in range(top, bottom + 1):
        border_indices.add((row, left))
        border_indices.add((row, right))
    for col in range(left, right + 1):
        border_indices.add((top, col))
        border_indices.add((bottom, col))
    for row, col in border_indices:
        grid[row][col] = value

def p(grid):
    corners = find_corners_with_value(grid)
    top, bottom, left, right = get_min_max_coordinates(corners)
    subgrid = extract_inner_section(grid, top, bottom, left, right)
    inner_section = [row[1:-1] for row in subgrid[1:-1]] if len(subgrid) > 2 and len(subgrid[0]) > 2 else []
    if inner_section:
        least_common_val = find_least_common_value([cell for row in inner_section for cell in row])
        paint_border_with_value(grid, top + 1, bottom - 1, left + 1, right - 1, least_common_val)
    return grid
