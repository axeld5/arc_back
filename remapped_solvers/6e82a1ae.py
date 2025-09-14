def explore_region(grid, start_row, start_col, visited):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    region = set()
    queue = [(start_row, start_col)]
    region.add((start_row, start_col))
    visited.add((start_row, start_col))
    while queue:
        current_row, current_col = queue.pop(0)
        for delta_row, delta_col in directions:
            new_row, new_col = (current_row + delta_row, current_col + delta_col)
            if 0 <= new_row < 10 and 0 <= new_col < 10 and (grid[new_row][new_col] == 5) and ((new_row, new_col) not in region):
                region.add((new_row, new_col))
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))
    return region

def update_grid_with_region_size(grid, region):
    region_size = len(region)
    value_to_set = 5 - region_size
    for row, col in region:
        grid[row][col] = value_to_set

def p(grid):
    visited = set()
    modified_grid = [row[:] for row in grid]
    for row in range(10):
        for col in range(10):
            if grid[row][col] == 5 and (row, col) not in visited:
                region = explore_region(grid, row, col, visited)
                update_grid_with_region_size(modified_grid, region)
    return modified_grid
