def find_zones(grid):
    rows, cols = (len(grid), len(grid[0]))
    visited = set()
    zones = []

    def valid_cell(x, y):
        return 0 <= x < rows and 0 <= y < cols and ((x, y) not in visited) and (grid[x][y] != 2)
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != 2 and (row, col) not in visited:
                zone_cells = []
                stack = [(row, col)]
                visited.add((row, col))
                empty_cell_count = 0
                while stack:
                    current_row, current_col = stack.pop()
                    zone_cells.append((current_row, current_col))
                    if grid[current_row][current_col] == 0:
                        empty_cell_count += 1
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        neighbor_row, neighbor_col = (current_row + dr, current_col + dc)
                        if valid_cell(neighbor_row, neighbor_col):
                            visited.add((neighbor_row, neighbor_col))
                            stack.append((neighbor_row, neighbor_col))
                zones.append((empty_cell_count, zone_cells))
    return zones

def update_grid_with_markers(grid, zones):
    largest_zone_size = max((zone[0] for zone in zones))
    smallest_zone_size = min((zone[0] for zone in zones))
    new_grid = [row[:] for row in grid]
    for empty_cell_count, zone_cells in zones:
        marker = 1 if empty_cell_count == largest_zone_size else 8 if empty_cell_count == smallest_zone_size else 0
        if marker:
            for row, col in zone_cells:
                if grid[row][col] == 0:
                    new_grid[row][col] = marker
    return new_grid

def p(grid, R=range):
    zones = find_zones(grid)
    return update_grid_with_markers(grid, zones)
