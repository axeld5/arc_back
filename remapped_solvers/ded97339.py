def fill_line(grid, value, start, end, fixed, is_row=True):
    if is_row:
        for col in range(start, end + 1):
            grid[fixed][col] = value
    else:
        for row in range(start, end + 1):
            grid[row][fixed] = value

def process_pairs(grid, positions, value):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r1, c1 = positions[i]
            r2, c2 = positions[j]
            if r1 == r2:
                fill_line(grid, value, min(c1, c2), max(c1, c2), r1, is_row=True)
            elif c1 == c2:
                fill_line(grid, value, min(r1, r2), max(r1, r2), c1, is_row=False)

def p(grid, number_range=range):
    solved_grid = [row[:] for row in grid]
    for num in number_range(1, 10):
        positions = [(row_idx, col_idx) for row_idx in range(len(grid)) for col_idx in range(len(grid[0])) if grid[row_idx][col_idx] == num]
        process_pairs(solved_grid, positions, num)
    return solved_grid
