from collections import deque

def p(grid):
    grid_size = 30
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    visited = {(row, col): False for row in range(grid_size) for col in range(grid_size)}
    largest_component = []
    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row][col] != 0 and (not visited[row, col]):
                current_component = explore_component(grid, row, col, visited, directions, grid_size)
                if len(current_component) > len(largest_component):
                    largest_component = current_component
    return extract_subgrid(grid, largest_component)

def explore_component(grid, start_row, start_col, visited, directions, grid_size):
    queue = deque([(start_row, start_col)])
    component_cells = []
    value = grid[start_row][start_col]
    visited[start_row, start_col] = True
    while queue:
        row, col = queue.popleft()
        component_cells.append((row, col))
        for dr, dc in directions:
            new_row, new_col = (row + dr, col + dc)
            if 0 <= new_row < grid_size and 0 <= new_col < grid_size and (not visited[new_row, new_col]) and (grid[new_row][new_col] == value):
                visited[new_row, new_col] = True
                queue.append((new_row, new_col))
    return component_cells

def extract_subgrid(grid, component_cells):
    if not component_cells:
        return []
    row_indices = [row for row, _ in component_cells]
    col_indices = [col for _, col in component_cells]
    min_row, max_row = (min(row_indices), max(row_indices) + 1)
    min_col, max_col = (min(col_indices), max(col_indices) + 1)
    return [list(grid[row][min_col:max_col]) for row in range(min_row, max_row)]
