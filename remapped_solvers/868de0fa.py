from collections import deque
from typing import List

def mark_connected_component(grid: List[List[int]], start_row: int, start_col: int, height: int, visited: List[List[bool]]) -> None:
    original_value = grid[start_row][start_col]
    component_queue = deque([(start_row, start_col)])
    component_cells = [(start_row, start_col)]
    visited[start_row][start_col] = True
    while component_queue:
        row, col = component_queue.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row, neighbor_col = (row + delta_row, col + delta_col)
            if 0 <= neighbor_row < height and 0 <= neighbor_col < height and (not visited[neighbor_row][neighbor_col]) and (grid[neighbor_row][neighbor_col] == original_value):
                visited[neighbor_row][neighbor_col] = True
                component_queue.append((neighbor_row, neighbor_col))
                component_cells.append((neighbor_row, neighbor_col))
    return component_cells

def update_component(grid: List[List[int]], component_cells: List[tuple], replacement_value: int) -> None:
    for row, col in component_cells:
        grid[row][col] = replacement_value

def p(grid: List[List[int]]) -> List[List[int]]:
    height = len(grid)
    visited = [[False] * height for _ in range(height)]
    for row in range(height):
        for col in range(height):
            if visited[row][col]:
                continue
            component_cells = mark_connected_component(grid, row, col, height, visited)
            if not component_cells:
                continue
            component_rows = [r for r, _ in component_cells]
            component_cols = [c for _, c in component_cells]
            row_span = max(component_rows) - min(component_rows) + 1
            col_span = max(component_cols) - min(component_cols) + 1
            if row_span == col_span and row_span * col_span == len(component_cells):
                replacement_value = 2 if row_span % 2 == 0 else 7
                update_component(grid, component_cells, replacement_value)
    return grid
