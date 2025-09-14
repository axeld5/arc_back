def mark_path_as_3(grid, row, col, delta_row, delta_col, steps):
    for step in range(1, steps):
        grid[row + step * delta_row][col + step * delta_col] = 3

def find_eights(grid):
    eight_positions = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 8:
                eight_positions.append((row, col))
    return eight_positions

def process_grid(grid):
    num_rows, num_cols = (len(grid), len(grid[0]))
    processed_grid = [row[:] for row in grid]
    eight_positions = find_eights(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for start_row, start_col in eight_positions:
        for delta_row, delta_col in directions:
            step = 1
            while 0 <= start_row + step * delta_row < num_rows and 0 <= start_col + step * delta_col < num_cols:
                if grid[start_row + step * delta_row][start_col + step * delta_col] == 8:
                    mark_path_as_3(processed_grid, start_row, start_col, delta_row, delta_col, step)
                    break
                step += 1
    return processed_grid

def p(j):
    return process_grid(j)
