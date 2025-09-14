def p(grid):
    GRID_SIZE = 21
    cell_positions = get_non_zero_positions(grid, GRID_SIZE)
    min_row, max_row, min_col, max_col = get_bounding_box(cell_positions)
    rectangle = extract_subgrid(grid, min_row, max_row, min_col, max_col)
    unique_columns = find_unique_columns(rectangle)
    filtered_grid = filter_columns(rectangle, unique_columns)
    distinct_rows = remove_duplicate_rows(filtered_grid)
    return distinct_rows

def get_non_zero_positions(grid, size):
    return [(r, c) for r in range(size) for c in range(size) if grid[r][c] != 0]

def get_bounding_box(positions):
    min_row = min((r for r, _ in positions))
    max_row = max((r for r, _ in positions))
    min_col = min((c for _, c in positions))
    max_col = max((c for _, c in positions))
    return (min_row, max_row, min_col, max_col)

def extract_subgrid(grid, min_row, max_row, min_col, max_col):
    return [grid[r][min_col:max_col + 1] for r in range(min_row, max_row + 1)]

def find_unique_columns(subgrid):
    unique_indices = set()
    for row in subgrid:
        changing_points = {0}
        for j in range(1, len(row)):
            if row[j] != row[j - 1]:
                changing_points.add(j)
        unique_indices.update(changing_points)
    return sorted(unique_indices)

def filter_columns(subgrid, column_indices):
    return [[row[col_idx] for col_idx in column_indices] for row in subgrid]

def remove_duplicate_rows(subgrid):
    distinct_rows = []
    i = 0
    while i < len(subgrid):
        distinct_rows.append(subgrid[i])
        j = i + 1
        while j < len(subgrid) and subgrid[j] == subgrid[i]:
            j += 1
        i = j
    return distinct_rows
