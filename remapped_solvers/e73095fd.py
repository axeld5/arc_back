def get_neighbors(i, j):
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

def is_in_bounds(i, j, height, width):
    return 0 <= i < height and 0 <= j < width

def find_connected_component(start, grid, visited):
    color = grid[start[0]][start[1]]
    queue = [start]
    component = set([start])
    while queue:
        i, j = queue.pop(0)
        for ni, nj in get_neighbors(i, j):
            if is_in_bounds(ni, nj, len(grid), len(grid[0])) and (ni, nj) not in component and (grid[ni][nj] == color):
                component.add((ni, nj))
                queue.append((ni, nj))
    return component

def find_zero_blocks(grid):
    visited = set()
    zero_blocks = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0 and (i, j) not in visited:
                block = find_connected_component((i, j), grid, visited)
                visited.update(block)
                zero_blocks.append(block)
    return zero_blocks

def is_surrounded_by_fives(block, fives_positions, grid_dimensions):
    x_values = [x for x, _ in block]
    y_values = [y for _, y in block]
    top, bottom = (min(x_values), max(x_values))
    left, right = (min(y_values), max(y_values))
    if len(block) != (bottom - top + 1) * (right - left + 1):
        return False
    x_border, y_border, a_border, d_border = (top - 1, left - 1, bottom + 1, right + 1)
    border_positions = set()
    for i in range(x_border, a_border + 1):
        if is_in_bounds(i, y_border, *grid_dimensions):
            border_positions.add((i, y_border))
        if is_in_bounds(i, d_border, *grid_dimensions):
            border_positions.add((i, d_border))
    for j in range(y_border, d_border + 1):
        if is_in_bounds(x_border, j, *grid_dimensions):
            border_positions.add((x_border, j))
        if is_in_bounds(a_border, j, *grid_dimensions):
            border_positions.add((a_border, j))
    corners = [(x_border, y_border), (x_border, d_border), (a_border, y_border), (a_border, d_border)]
    extended_border = set()
    for i, j in corners:
        for ni, nj in get_neighbors(i, j):
            if is_in_bounds(ni, nj, *grid_dimensions):
                extended_border.add((ni, nj))
    if not extended_border - border_positions & fives_positions:
        return True
    return False

def p(grid, K=range):
    height, width = (len(grid), len(grid[0]))
    zero_blocks = find_zero_blocks(grid)
    fives_positions = {(i, j) for i in K(height) for j in K(width) if grid[i][j] == 5}
    transformed_positions = set()
    for block in zero_blocks:
        if is_surrounded_by_fives(block, fives_positions, (height, width)):
            transformed_positions.update(block)
    for i, j in transformed_positions:
        grid[i][j] = 4
    return grid
