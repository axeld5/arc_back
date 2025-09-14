from collections import deque

def get_filled_cells(grid):
    height = len(grid)
    filled_positions = [(i, j) for i, row in enumerate(grid) for j, value in enumerate(row) if value == 4]
    filled_cells = set()
    for start in filled_positions:
        for end in filled_positions:
            if start[0] == end[0]:
                for column in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                    filled_cells.add((start[0], column))
            elif start[1] == end[1]:
                for row in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                    filled_cells.add((row, start[1]))
    return filled_cells

def mark_empty_cells(grid, filled_cells):
    for i, j in filled_cells:
        if grid[i][j] == 0:
            grid[i][j] = -1

def find_connected_components(grid):
    height = len(grid)
    visited = [[0] * height for _ in range(height)]
    components = []
    for i in range(height):
        for j in range(height):
            if visited[i][j] or grid[i][j] == 0:
                continue
            queue = deque([(i, j)])
            visited[i][j] = 1
            component = {(i, j)}
            while queue:
                curr_row, curr_col = queue.popleft()
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor_row, neighbor_col = (curr_row + d_row, curr_col + d_col)
                    if 0 <= neighbor_row < height and 0 <= neighbor_col < height and (not visited[neighbor_row][neighbor_col]) and (grid[neighbor_row][neighbor_col] != 0):
                        visited[neighbor_row][neighbor_col] = 1
                        queue.append((neighbor_row, neighbor_col))
                        component.add((neighbor_row, neighbor_col))
            components.append(component)
    return components

def apply_inside_region(grid, components):
    height = len(grid)
    fill_cells = set()
    for component in components:
        min_row = min((i for i, j in component))
        min_col = min((j for i, j in component))
        max_row = max((i for i, j in component))
        max_col = max((j for i, j in component))
        if min_row + 1 <= max_row - 1 and min_col + 1 <= max_col - 1:
            for i in range(min_row + 1, max_row):
                for j in range(min_col + 1, max_col):
                    fill_cells.add((i, j))
    for i, j in fill_cells:
        grid[i][j] = 2

def p(grid):
    height = 10
    grid_copy = [list(row) for row in grid]
    filled_cells = get_filled_cells(grid_copy)
    mark_empty_cells(grid_copy, filled_cells)
    components = find_connected_components(grid_copy)
    apply_inside_region(grid_copy, components)
    for i in range(height):
        for j in range(height):
            if grid_copy[i][j] == -1:
                grid_copy[i][j] = 0
    return grid_copy
