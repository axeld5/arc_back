def is_isolated(grid, row, col, max_height, max_width):
    if grid[row][col] == 0:
        return False
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in neighbors:
        neighbor_row, neighbor_col = (row + dr, col + dc)
        if 0 <= neighbor_row < max_height and 0 <= neighbor_col < max_width:
            if grid[neighbor_row][neighbor_col] != 0:
                return False
    return True

def solve(grid):
    height = len(grid)
    width = len(grid[0])
    return [[1 if is_isolated(grid, row, col, height, width) else grid[row][col] for col in range(width)] for row in range(height)]
p = solve
