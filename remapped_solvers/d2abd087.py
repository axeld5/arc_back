from collections import deque

def p(grid, K=range):
    GRID_SIZE = 10
    visited = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    size_6_group = []
    other_groups = []
    for row in K(GRID_SIZE):
        for col in K(GRID_SIZE):
            if visited[row][col] or grid[row][col] == 0:
                continue
            cell_value = grid[row][col]
            connected_group = explore_group(grid, visited, cell_value, row, col, GRID_SIZE)
            if len(connected_group) == 6:
                size_6_group.extend(connected_group)
            else:
                other_groups.extend(connected_group)
    return mark_groups(size_6_group, other_groups, GRID_SIZE, grid)

def explore_group(grid, visited, cell_value, start_row, start_col, grid_size):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque([(start_row, start_col)])
    group_cells = []
    visited[start_row][start_col] = 1
    while queue:
        current_row, current_col = queue.popleft()
        group_cells.append((current_row, current_col))
        for direction in directions:
            new_row, new_col = (current_row + direction[0], current_col + direction[1])
            if 0 <= new_row < grid_size and 0 <= new_col < grid_size and (not visited[new_row][new_col]) and (grid[new_row][new_col] == cell_value):
                visited[new_row][new_col] = 1
                queue.append((new_row, new_col))
    return group_cells

def mark_groups(size_6_group, other_groups, grid_size, original_grid):
    marked_grid = [[0] * grid_size for _ in range(grid_size)]
    size_6_set = set(size_6_group)
    other_set = set(other_groups)
    for row in range(grid_size):
        for col in range(grid_size):
            cell = (row, col)
            if cell in size_6_set:
                marked_grid[row][col] = 2
            elif cell in other_set:
                marked_grid[row][col] = 1
            else:
                marked_grid[row][col] = original_grid[row][col]
    return marked_grid
