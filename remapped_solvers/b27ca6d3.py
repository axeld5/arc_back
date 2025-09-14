def transpose(grid):
    return [[grid[row][col] for row in range(len(grid))] for col in range(len(grid[0]))]

def mark_blocked_areas(grid, row, col):
    if grid[row][col] != 2 or grid[row][col + 1] != 2:
        return
    for r in range(max(0, row - 1), min(len(grid), row + 2)):
        for c in range(max(0, col - 1), min(len(grid[0]), col + 3)):
            if grid[r][c] != 2:
                grid[r][c] = 3

def process_grid(grid):
    for row in range(len(grid)):
        for col in range(len(grid[0]) - 1):
            mark_blocked_areas(grid, row, col)

def p(grid):
    process_grid(grid)
    transposed_grid = transpose(grid)
    process_grid(transposed_grid)
    return transpose(transposed_grid)
