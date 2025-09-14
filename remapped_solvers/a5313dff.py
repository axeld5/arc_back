def mark_border_regions(grid, marked, queue):
    size = len(grid)
    for row in range(size):
        for col in range(size):
            if row * col == 0 or row == size - 1 or col == size - 1:
                if grid[row][col] == 0:
                    marked[row][col] = 1
                    queue.append((row, col))

def expand_water_regions(grid, marked, queue):
    size = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        current_row, current_col = queue.pop(0)
        for dr, dc in directions:
            new_row, new_col = (current_row + dr, current_col + dc)
            if 0 <= new_row < size and 0 <= new_col < size and (grid[new_row][new_col] == 0) and (not marked[new_row][new_col]):
                marked[new_row][new_col] = 1
                queue.append((new_row, new_col))

def create_filled_grid(grid, marked):
    size = len(grid)
    filled_grid = [[grid[row][col] if grid[row][col] != 0 or marked[row][col] else 1 for col in range(size)] for row in range(size)]
    return filled_grid

def p(grid):
    size = len(grid)
    marked = [[0] * size for _ in range(size)]
    queue = []
    mark_border_regions(grid, marked, queue)
    expand_water_regions(grid, marked, queue)
    return create_filled_grid(grid, marked)
