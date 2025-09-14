from collections import deque
from typing import List, Tuple, Set

def p(grid: List[List[int]], range_func=range, min_func=min, max_func=max, next_func=next) -> List[List[int]]:
    height = len(grid)
    width = len(grid[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited_points: Set[Tuple[int, int]] = set()
    clusters = []

    def find_clusters():
        for y in range_func(height):
            for x in range_func(width):
                if (y, x) in visited_points or grid[y][x] == 0:
                    continue
                cluster = []
                queue = deque([(y, x)])
                visited_points.add((y, x))
                while queue:
                    row, col = queue.pop()
                    cluster.append((grid[row][col], (row, col)))
                    for dx, dy in directions:
                        nx, ny = (row + dx, col + dy)
                        if 0 <= nx < height and 0 <= ny < width and ((nx, ny) not in visited_points) and (grid[nx][ny] != 0):
                            visited_points.add((nx, ny))
                            queue.append((nx, ny))
                clusters.append(cluster)
    find_clusters()
    target_cluster = next_func((cluster for cluster in clusters if cluster[0][0] == 5))
    non_target_cluster = next_func((cluster for cluster in clusters if cluster is not target_cluster))
    target_rows = [point[1][0] for point in target_cluster]
    target_cols = [point[1][1] for point in target_cluster]
    target_midpoint = (min_func(target_rows) + (max_func(target_rows) - min_func(target_rows)) // 2, min_func(target_cols) + (max_func(target_cols) - min_func(target_cols)) // 2)
    min_non_target_row = min_func((point[1][0] for point in non_target_cluster))
    min_non_target_col = min_func((point[1][1] for point in non_target_cluster))
    translated_points = [(value, (row - min_non_target_row + target_midpoint[0] - 1, col - min_non_target_col + target_midpoint[1] - 1)) for value, (row, col) in non_target_cluster]
    for value, (row, col) in translated_points:
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = value
    return grid
