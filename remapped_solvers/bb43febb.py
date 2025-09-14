def is_valid_position(grid, row, col):
    if grid[row][col] != 5:
        return False
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in neighbors:
        neighbor_row, neighbor_col = (row + dr, col + dc)
        if not (0 <= neighbor_row < 10 and 0 <= neighbor_col < 10):
            return False
        if grid[neighbor_row][neighbor_col] != 5:
            return False
    return True

def process_grid(grid):
    processed_grid = []
    for row in range(10):
        processed_row = []
        for col in range(10):
            if is_valid_position(grid, row, col):
                processed_row.append(2)
            else:
                processed_row.append(grid[row][col])
        processed_grid.append(processed_row)
    return processed_grid

def p(grid):
    return process_grid(grid)
