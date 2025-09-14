from collections import defaultdict
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def p(grid, R=range):
    grid_size = 21
    in_bounds = lambda row, col: 0 <= row < grid_size and 0 <= col < grid_size
    color_cells = defaultdict(list)
    for row in R(grid_size):
        for col in R(grid_size):
            if grid[row][col]:
                color_cells[grid[row][col]].append((row, col))

    def bounding_box(positions):
        rows = [row for row, _ in positions]
        cols = [col for _, col in positions]
        return (min(rows), max(rows), min(cols), max(cols))
    potential_patterns = []
    for color, positions in color_cells.items():
        min_row, max_row, min_col, max_col = bounding_box(positions)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        if height <= 3 and width <= 3:
            pattern_positions = {(row - min_row, col - min_col) for row, col in positions}
            potential_patterns.append((len(positions), color, min_row, min_col, pattern_positions))
    potential_patterns.sort(reverse=True)
    _, main_color, start_row, start_col, main_pattern = potential_patterns[0]
    neighboring_max_color = {}
    for delta_row, delta_col in DIRECTIONS:
        search_row, search_col = (start_row + 4 * delta_row, start_col + 4 * delta_col)
        matching_counts = defaultdict(int)
        for offset_row, offset_col in main_pattern:
            neighbor_row, neighbor_col = (search_row + offset_row, search_col + offset_col)
            if in_bounds(neighbor_row, neighbor_col) and grid[neighbor_row][neighbor_col]:
                matching_counts[grid[neighbor_row][neighbor_col]] += 1
        if matching_counts:
            neighboring_max_color[delta_row, delta_col] = max(matching_counts.items(), key=lambda x: x[1])[0]
    neighboring_max_color[0, 0] = main_color
    output_grid = [[0] * grid_size for _ in R(grid_size)]

    def fill_pattern(base_row, base_col, color):
        for offset_row, offset_col in main_pattern:
            target_row, target_col = (base_row + offset_row, base_col + offset_col)
            if in_bounds(target_row, target_col):
                output_grid[target_row][target_col] = color
    for (delta_row, delta_col), color in neighboring_max_color.items():
        row, col = (start_row, start_col)
        if (delta_row, delta_col) == (0, 0):
            fill_pattern(row, col, color)
            continue
        while True:
            row += 4 * delta_row
            col += 4 * delta_col
            fill_pattern(row, col, color)
            if row < -5 or row > grid_size or col < -5 or (col > grid_size):
                break
    return output_grid
