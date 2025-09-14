from collections import Counter, deque

def get_8_neighbors(i, j):
    return ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1))

def p(grid):
    height, width = (len(grid), len(grid[0]))

    def get_palette(element):
        if isinstance(element, (list, tuple)) and isinstance(element[0], (list, tuple)):
            return {color for row in element for color in row}
        return {color for color, _ in element}

    def most_common_color(grid):
        return Counter((color for row in grid for color in row)).most_common(1)[0][0]

    def find_objects(grid):
        background = most_common_color(grid)
        visited = [[0] * width for _ in range(height)]
        objects = []
        for start_i in range(height):
            for start_j in range(width):
                value = grid[start_i][start_j]
                if value == background or visited[start_i][start_j]:
                    continue
                queue = deque([(start_i, start_j)])
                visited[start_i][start_j] = 1
                component = {(value, (start_i, start_j))}
                while queue:
                    i, j = queue.popleft()
                    for ni, nj in get_8_neighbors(i, j):
                        if 0 <= ni < height and 0 <= nj < width and (not visited[ni][nj]) and (grid[ni][nj] != background):
                            visited[ni][nj] = 1
                            queue.append((ni, nj))
                            component.add((grid[ni][nj], (ni, nj)))
                objects.append(component)
        return objects

    def normalize_object(obj):
        min_i = min((i for _, (i, _) in obj))
        min_j = min((j for _, (_, j) in obj))
        return set(((color, (i - min_i, j - min_j)) for color, (i, j) in obj))

    def upscale_object(obj, scale):
        return set(((color, (i * scale + di, j * scale + dj)) for color, (i, j) in obj for di in range(scale) for dj in range(scale)))

    def paint_object(grid, obj):
        new_grid = [list(row) for row in grid]
        for color, (i, j) in obj:
            if 0 <= i < height and 0 <= j < width:
                new_grid[i][j] = color
        return new_grid

    def get_positions_of_color(grid, value):
        return {(i, j) for i in range(height) for j in range(width) if grid[i][j] == value}

    def fill_positions(grid, value, positions):
        new_grid = [list(row) for row in grid]
        for i, j in positions:
            new_grid[i][j] = value
        return new_grid
    objects = find_objects(grid)
    smallest_object = min(objects, key=lambda obj: len(get_palette(obj)))
    upscaled_object = upscale_object(normalize_object(smallest_object), 4)
    painted_grid = paint_object(grid, upscaled_object)
    return fill_positions(painted_grid, 5, get_positions_of_color(grid, 5))
