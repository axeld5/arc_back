from collections import deque

def p(grid, range_=range):
    grid_size = 10
    visited = [[0] * grid_size for _ in range_(grid_size)]
    regions = []

    def get_valid_neighbors(x, y, current_value):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_x, new_y = (x + dx, y + dy)
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size and (not visited[new_x][new_y]) and (grid[new_x][new_y] == current_value):
                yield (new_x, new_y)
    for row in range_(grid_size):
        for col in range_(grid_size):
            if grid[row][col] == 0 or visited[row][col]:
                continue
            current_value = grid[row][col]
            region_cells = []
            cells_to_visit = deque([(row, col)])
            visited[row][col] = 1
            while cells_to_visit:
                x, y = cells_to_visit.popleft()
                region_cells.append((x, y))
                for neighbor_x, neighbor_y in get_valid_neighbors(x, y, current_value):
                    visited[neighbor_x][neighbor_y] = 1
                    cells_to_visit.append((neighbor_x, neighbor_y))
            regions.append((region_cells, current_value))
    max_region_size = max((len(region) for region, _ in regions))
    largest_regions = [(cells, value) for cells, value in regions if len(cells) == max_region_size]
    largest_regions.sort(key=lambda region: min((col for _, col in region[0])))
    most_frequent_values = [value for _, value in largest_regions]
    return [most_frequent_values[:] for _ in range_(max_region_size)]
