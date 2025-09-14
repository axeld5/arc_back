from collections import deque

def find_connected_component(grid, visited, start_row, start_col):
    rows, cols = (len(grid), len(grid[0]))
    queue = deque([(start_row, start_col)])
    component = []
    visited[start_row][start_col] = True
    while queue:
        row, col = queue.popleft()
        component.append((grid[row][col], (row, col)))
        for new_row, new_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if not visited[new_row][new_col] and grid[new_row][new_col] != 0:
                    visited[new_row][new_col] = True
                    queue.append((new_row, new_col))
    return component

def find_largest_component(grid):
    visited = [[False] * 9 for _ in range(9)]
    components = []
    for row in range(9):
        for col in range(9):
            if not visited[row][col] and grid[row][col] != 0:
                component = find_connected_component(grid, visited, row, col)
                components.append(component)
    largest_component = max(components, key=lambda comp: sum((value == 1 for value, _ in comp)))
    return largest_component

def extract_subgrid_from_component(grid, largest_component):
    row_indices = [row for _, (row, _) in largest_component]
    col_indices = [col for _, (_, col) in largest_component]
    min_row, max_row = (min(row_indices), max(row_indices))
    min_col, max_col = (min(col_indices), max(col_indices))
    subgrid = [grid[row][min_col:max_col + 1] for row in range(min_row, max_row + 1)]
    return subgrid

def p(grid):
    largest_component = find_largest_component(grid)
    return extract_subgrid_from_component(grid, largest_component)
