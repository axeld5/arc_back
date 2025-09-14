from collections import deque

def identify_objects(grid):
    height, width = (len(grid), len(grid[0]))
    visited = [[False] * width for _ in range(height)]
    objects = []
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 0 or visited[i][j]:
                continue
            queue = deque([(i, j)])
            visited[i][j] = True
            current_object = {(i, j)}
            while queue:
                x, y = queue.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = (x + dx, y + dy)
                        if 0 <= new_x < height and 0 <= new_y < width and (grid[new_x][new_y] != 0) and (not visited[new_x][new_y]):
                            visited[new_x][new_y] = True
                            queue.append((new_x, new_y))
                            current_object.add((new_x, new_y))
            objects.append(current_object)
    return objects

def get_bounding_box(obj):
    rows, cols = zip(*obj)
    return (min(rows), min(cols), max(rows), max(cols))

def extract_object(grid, bounding_box):
    min_row, min_col, max_row, max_col = bounding_box
    return [row[min_col:max_col + 1] for row in grid[min_row:max_row + 1]]

def downscale_grid(grid, factor):
    return [[row[j] for j in range(0, len(row), factor)] for row in grid[::factor]]

def p(grid):
    objects = identify_objects(grid)
    largest_object = max(objects, key=len)
    smallest_object = min(objects, key=len)
    largest_bounding_box = get_bounding_box(largest_object)
    smallest_bounding_box = get_bounding_box(smallest_object)
    largest_grid = extract_object(grid, largest_bounding_box)
    smallest_grid = extract_object(grid, smallest_bounding_box)
    downscale_factor = max(1, len(largest_grid[0]) // len(smallest_grid[0]))
    downscaled_largest_grid = downscale_grid(largest_grid, downscale_factor)
    zero_positions = {(i, j) for i, row in enumerate(downscaled_largest_grid) for j, value in enumerate(row) if value == 0}
    smallest_grid_copy = [row[:] for row in smallest_grid]
    for i, j in zero_positions:
        if 0 <= i < len(smallest_grid_copy) and 0 <= j < len(smallest_grid_copy[i]):
            smallest_grid_copy[i][j] = 0
    return smallest_grid_copy
