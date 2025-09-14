def get_cut_row(grid, height, width):
    for row in range(height):
        if all((grid[row][col] for col in range(width))):
            return row
    return None

def determine_vertical_cut(grid, height, width, cut_row):
    for col in range(width):
        if grid[cut_row][col]:
            return col
    return None

def collect_columns_to_match(grid, width, cut_row, vertical_cut):
    match_columns = set()
    for col in range(width):
        if grid[cut_row][col] == grid[cut_row][vertical_cut]:
            match_columns.add(col)
    return match_columns

def assign_colors(row_values, match_columns):
    color_map = {}
    for col, value in enumerate(row_values):
        if value:
            if col in match_columns:
                color_map[value] = 0
            else:
                color_map.setdefault(value, 1)
            if len(color_map) == 2:
                break
    if 1 not in color_map.values():
        color_map[0] = color_map.get(0, 0)
    return color_map

def solve_grid(grid):
    height = len(grid)
    width = len(grid[0])
    output_grid = [[0] * width for _ in range(height)]
    cut_row = get_cut_row(grid, height, width)
    if cut_row is None:
        return output_grid
    vertical_cut = determine_vertical_cut(grid, height, width, cut_row)
    match_columns = collect_columns_to_match(grid, width, cut_row, vertical_cut)
    for row in range(height):
        if not any(grid[row]):
            continue
        color_map = assign_colors(grid[row], match_columns)
        inverse_map = {v: k for k, v in color_map.items()}
        for col in range(width):
            if col in match_columns:
                output_grid[row][col] = inverse_map[0]
            else:
                output_grid[row][col] = inverse_map.get(1, inverse_map[0])
    return output_grid

def p(I, A=range):
    return solve_grid(I)
