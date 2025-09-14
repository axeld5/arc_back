from collections import deque

def find_clusters(grid):
    grid_size = len(grid)
    visited = [[False] * grid_size for _ in range(grid_size)]
    clusters = []
    for row in range(grid_size):
        for col in range(grid_size):
            if visited[row][col] or grid[row][col] == 0:
                continue
            cell_value = grid[row][col]
            cluster_cells_row_indices, cluster_cells_col_indices = breadth_first_search(grid, grid_size, visited, row, col, cell_value)
            cluster_center = calculate_center(cluster_cells_row_indices, cluster_cells_col_indices, cell_value)
            clusters.append(cluster_center)
    return clusters

def breadth_first_search(grid, grid_size, visited, start_row, start_col, cell_value):
    queue = deque([(start_row, start_col)])
    row_indices, col_indices = ([], [])
    while queue:
        curr_row, curr_col = queue.popleft()
        if visited[curr_row][curr_col] or grid[curr_row][curr_col] != cell_value:
            continue
        visited[curr_row][curr_col] = True
        row_indices.append(curr_row)
        col_indices.append(curr_col)
        for delta_row, delta_col in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_row, neighbor_col = (curr_row + delta_row, curr_col + delta_col)
            if 0 <= neighbor_row < grid_size and 0 <= neighbor_col < grid_size:
                queue.append((neighbor_row, neighbor_col))
    return (row_indices, col_indices)

def calculate_center(row_indices, col_indices, value):
    min_row, max_row = (min(row_indices), max(row_indices))
    min_col, max_col = (min(col_indices), max(col_indices))
    center_row = min_row + (max_row - min_row) // 2
    center_col = min_col + (max_col - min_col) // 2
    return (value, center_row, center_col)

def assign_nearest_cluster(grid, clusters):
    grid_size = len(grid)
    cluster_centers = [(center_row, center_col) for _, center_row, center_col in clusters]
    for row in range(grid_size):
        for col in range(grid_size):
            cluster_index = find_unique_nearest_cluster((row, col), cluster_centers)
            if cluster_index == -1:
                continue
            cluster_value, center_row, center_col = clusters[cluster_index]
            if max(abs(row - center_row), abs(col - center_col)) % 2 == 0:
                grid[row][col] = cluster_value
    return grid

def find_unique_nearest_cluster(position, cluster_centers):
    row, col = position
    distances = [abs(row - center_row) + abs(col - center_col) for center_row, center_col in cluster_centers]
    min_distance = min(distances)
    if distances.count(min_distance) != 1:
        return -1
    return distances.index(min_distance)

def p(grid):
    clusters = find_clusters(grid)
    solved_grid = assign_nearest_cluster(grid, clusters)
    return solved_grid
