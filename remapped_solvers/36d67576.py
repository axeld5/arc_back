from collections import deque

def rotate_and_flip_positions(positions, height, width, rotations, flip):

    def rotate(p, h, w, k):
        if k % 4 == 0:
            return (h, w, [(r, c) for r, c in p])
        elif k % 4 == 1:
            return (w, h, [(c, h - 1 - r) for r, c in p])
        elif k % 4 == 2:
            return (h, w, [(h - 1 - r, w - 1 - c) for r, c in p])
        else:
            return (w, h, [(w - 1 - c, r) for r, c in p])

    def flip_positions(p, h, w):
        return (h, w, [(r, w - 1 - c) for r, c in p])
    if flip:
        height, width, positions = flip_positions(positions, height, width)
    return rotate(positions, height, width, rotations)

def find_and_transform_component(grid, height, width, start_row, start_col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    queue = deque([(start_row, start_col)])
    seen = set(queue)
    component = []
    while queue:
        x, y = queue.popleft()
        component.append((x, y))
        for dx, dy in directions:
            nx, ny = (x + dx, y + dy)
            if 0 <= nx < height and 0 <= ny < width and grid[nx][ny] and ((nx, ny) not in seen):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return component

def find_target_component(grid, height, width):
    seen = [[False] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            if grid[r][c] and (not seen[r][c]):
                component = find_and_transform_component(grid, height, width, r, c)
                component_types = {grid[x][y] for x, y in component}
                if {1, 2, 3, 4}.issubset(component_types):
                    return component
    return None

def transform_grid(grid):
    height, width = (len(grid), len(grid[0]))
    original_grid = [row[:] for row in grid]
    target_component = find_target_component(original_grid, height, width)
    if not target_component:
        return original_grid
    min_r = min((r for r, _ in target_component))
    min_c = min((c for _, c in target_component))

    def get_component_by_type(type_id):
        return [(r - min_r, c - min_c) for r, c in target_component if grid[r][c] == type_id]
    red_positions = get_component_by_type(2)
    yellow_positions = get_component_by_type(4)
    blue_positions = get_component_by_type(1)
    green_positions = get_component_by_type(3)
    block_height = max((r for r, _ in red_positions + yellow_positions)) + 1
    block_width = max((c for _, c in red_positions + yellow_positions)) + 1
    for flip in (False, True):
        for rotations in range(4):
            L, W, red_positions_rotated = rotate_and_flip_positions(red_positions, block_height, block_width, rotations, flip)
            _, _, yellow_positions_rotated = rotate_and_flip_positions(yellow_positions, block_height, block_width, rotations, flip)
            _, _, blue_positions_rotated = rotate_and_flip_positions(blue_positions, block_height, block_width, rotations, flip)
            _, _, green_positions_rotated = rotate_and_flip_positions(green_positions, block_height, block_width, rotations, flip)
            for r in range(height - L + 1):
                for c in range(width - W + 1):
                    if all((original_grid[r + u][c + w] == 2 for u, w in red_positions_rotated)) and all((original_grid[r + u][c + w] == 4 for u, w in yellow_positions_rotated)) and (not any((original_grid[r + u][c + w] in (2, 4) for u, w in blue_positions_rotated + green_positions_rotated))):
                        for u, w in blue_positions_rotated:
                            original_grid[r + u][c + w] = 1
                        for u, w in green_positions_rotated:
                            original_grid[r + u][c + w] = 3
    return original_grid

def p(grid):
    return transform_grid(grid)
