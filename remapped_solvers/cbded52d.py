from collections import Counter

def p(grid, range_fn=range, min_fn=min, max_fn=max):
    grid_size = len(grid)
    most_common_value = find_most_common_value(grid)
    isolated_cells = find_isolated_cells(grid, most_common_value, grid_size)
    new_centers = {}
    for x1, y1, value in isolated_cells:
        for x2, y2, _ in isolated_cells:
            if x1 == x2 or y1 == y2:
                cells_in_line = get_cells_in_line(x1, y1, x2, y2, range_fn, min_fn, max_fn)
                center_x, center_y = find_center_of_line(cells_in_line, min_fn, max_fn)
                new_centers[center_x, center_y] = value
    update_grid_with_centers(grid, new_centers, grid_size)
    return grid

def find_most_common_value(grid):
    return Counter((value for row in grid for value in row)).most_common(1)[0][0]

def find_isolated_cells(grid, common_value, grid_size):
    isolated_cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row][col] != common_value and grid[row][col] >= 0:
                if is_isolated(grid, row, col, grid_size):
                    isolated_cells.append((row, col, grid[row][col]))
    return isolated_cells

def is_isolated(grid, row, col, size):
    value = grid[row][col]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d_row, d_col in directions:
        new_row, new_col = (row + d_row, col + d_col)
        if 0 <= new_row < size and 0 <= new_col < size and (grid[new_row][new_col] == value):
            return False
    return True

def get_cells_in_line(x1, y1, x2, y2, range_fn, min_fn, max_fn):
    if x1 == x2:
        return [(x1, col) for col in range_fn(min_fn(y1, y2), max_fn(y1, y2) + 1)]
    else:
        return [(row, y1) for row in range_fn(min_fn(x1, x2), max_fn(x1, x2) + 1)]

def find_center_of_line(line, min_fn, max_fn):
    min_x = min_fn((row for row, _ in line))
    max_x = max_fn((row for row, _ in line))
    min_y = min_fn((col for _, col in line))
    max_y = max_fn((col for _, col in line))
    center_x = min_x + (max_x - min_x) // 2
    center_y = min_y + (max_y - min_y) // 2
    return (center_x, center_y)

def update_grid_with_centers(grid, new_centers, grid_size):
    for (i, j), value in new_centers.items():
        if 0 <= i < grid_size and 0 <= j < grid_size:
            grid[i][j] = value
