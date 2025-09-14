def p(I, R=range):

    def get_color_counts(grid):
        color_count = {}
        for row in grid:
            for color in row:
                if color:
                    color_count[color] = color_count.get(color, 0) + 1
        return color_count

    def count_neighborhood_connections(grid, row, col, color):
        connection_count = 0
        for dr in R(-1, 2):
            for dc in R(-1, 2):
                nr, nc = (row + dr, col + dc)
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (grid[nr][nc] == color):
                    connection_count += 1
        return connection_count

    def find_connected_regions(grid, color):
        connected_count = 0
        for r in R(len(grid)):
            for c in R(len(grid[0])):
                if grid[r][c] == color:
                    if count_neighborhood_connections(grid, r, c, color) >= 4:
                        connected_count += 1
        return connected_count

    def get_primary_and_secondary_colors(colors):
        max_connected = -1
        primary_color = None
        for color in colors:
            connections = find_connected_regions(I, color)
            if connections > max_connected:
                max_connected = connections
                primary_color = color
        secondary_color = next((c for c in colors if c != primary_color), primary_color)
        return (primary_color, secondary_color)

    def calculate_3x3_transformed_grid(primary_color, secondary_color):
        rows = [r for r in R(len(I)) if any((I[r][c] == primary_color for c in R(len(I[r]))))]
        cols = [c for c in R(len(I[0])) if any((I[r][c] == primary_color for r in R(len(I))))]
        if not rows or not cols:
            return [[0] * 3 for _ in R(3)]
        row_min, row_max = (min(rows), max(rows))
        col_min, col_max = (min(cols), max(cols))
        height, width = (row_max - row_min + 1, col_max - col_min + 1)
        section_size = max(1, min(height // 3, width // 3))
        result_grid = [[0] * 3 for _ in R(3)]
        for x in R(3):
            for y in R(3):
                u, v = (row_min + x * section_size, col_min + y * section_size)
                total_cells = occupied_cells = 0
                for r in R(u, min(u + section_size, len(I))):
                    for c in R(v, min(v + section_size, len(I[0]))):
                        total_cells += 1
                        if I[r][c] == primary_color:
                            occupied_cells += 1
                if total_cells and occupied_cells * 2 >= total_cells:
                    result_grid[x][y] = secondary_color
        return result_grid
    color_counts = get_color_counts(I)
    if not color_counts:
        return [[0] * 3 for _ in R(3)]
    primary_color, secondary_color = get_primary_and_secondary_colors(color_counts.keys())
    return calculate_3x3_transformed_grid(primary_color, secondary_color)
