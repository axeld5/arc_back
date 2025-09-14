from collections import Counter

def rotate_clockwise(grid):
    rows, cols = (len(grid), len(grid[0]))
    return [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]

def rotate_counter_clockwise(grid):
    rows, cols = (len(grid), len(grid[0]))
    return [[grid[r][cols - 1 - c] for r in range(rows)] for c in range(cols)]

def normalize_view(grid):
    copy_grid = [row[:] for row in grid]
    unrotate = lambda x: x
    for _ in range(4):
        if all((cell == 2 for cell in copy_grid[0])):
            break
        copy_grid = rotate_clockwise(copy_grid)
        unrotate = lambda base, f=rotate_counter_clockwise, old=unrotate: old(f(base))
    return (copy_grid, unrotate)

def find_assembly_line_column_candidates(grid, start_row):
    rows, cols = (len(grid), len(grid[0]))
    candidates = Counter()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                diff = r - start_row
                candidates[c - diff] += 1
                candidates[c + diff] += 1
    return candidates

def place_items(grid, start_row, assembly_col):
    rows, cols = (len(grid), len(grid[0]))
    new_grid = [row[:] for row in grid]
    for c in range(cols):
        row_offset = start_row + abs(c - assembly_col)
        if 0 <= row_offset < rows:
            if grid[row_offset][c] != 8:
                new_grid[row_offset][c] = 3
            else:
                new_grid[row_offset][c] = 8
    return new_grid

def p(grid, _range=range):
    RELIC = 2
    GADGET = 8
    PLACED_GADGET = 3
    normalized_grid, unrotate = normalize_view(grid)
    start_row = 0
    while start_row < len(normalized_grid) and all((normalized_grid[start_row][col] == RELIC for col in _range(len(normalized_grid[0])))):
        start_row += 1
    column_counts = find_assembly_line_column_candidates(normalized_grid, start_row)
    if column_counts:
        optimal_column = max(column_counts, key=column_counts.get)
    else:
        optimal_column = len(normalized_grid[0]) // 2
    optimal_column = max(0, min(len(normalized_grid[0]) - 1, optimal_column))
    output_grid = place_items(normalized_grid, start_row, optimal_column)
    return unrotate(output_grid)
