def find_connected_components(grid, size):
    component_ids = [[-1] * size for _ in range(size)]
    connected_components = []
    component_count = 0
    for i in range(size):
        for j in range(size):
            if grid[i][j] != 0 or component_ids[i][j] != -1:
                continue
            queue = [(i, j)]
            component_ids[i][j] = component_count
            component_cells = [(i, j)]
            while queue:
                row, col = queue.pop()
                for new_row, new_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if 0 <= new_row < size and 0 <= new_col < size and (grid[new_row][new_col] == 0) and (component_ids[new_row][new_col] == -1):
                        component_ids[new_row][new_col] = component_count
                        queue.append((new_row, new_col))
                        component_cells.append((new_row, new_col))
            connected_components.append(component_cells)
            component_count += 1
    return (connected_components, component_ids)

def expand_connected_components(grid, connected_components, component_ids, size):
    unique_component_identifiers = set()
    for i in range(size):
        for j in range(size):
            if grid[i][j] != 1:
                continue
            for adj_row, adj_col in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= adj_row < size and 0 <= adj_col < size:
                    component_id = component_ids[adj_row][adj_col]
                    if component_id != -1:
                        unique_component_identifiers.add(component_id)
    for component_id in unique_component_identifiers:
        for row, col in connected_components[component_id]:
            grid[row][col] = 1

def p(grid):
    size = len(grid)
    connected_components, component_ids = find_connected_components(grid, size)
    expand_connected_components(grid, connected_components, component_ids, size)
    return grid
