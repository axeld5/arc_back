def get_bounding_box(grid):
    non_zero_rows = [row_index for row_index, row in enumerate(grid) for col, value in enumerate(row) if value != 0]
    non_zero_cols = [col for row_index, row in enumerate(grid) for col, value in enumerate(row) if value != 0]
    min_row, max_row = (min(non_zero_rows), max(non_zero_rows))
    min_col, max_col = (min(non_zero_cols), max(non_zero_cols))
    return (min_row, min_col, max_row, max_col)

def choose_direction(grid, bounding_box):
    min_row, min_col, max_row, max_col = bounding_box
    calculate_gaps = [(sum((1 for col in range(min_col, max_col + 1) if grid[min_row][col] == 0)), 0 if min_row > 0 else -1, 'U'), (sum((1 for col in range(min_col, max_col + 1) if grid[max_row][col] == 0)), 0 if max_row < len(grid) - 1 else -1, 'D'), (sum((1 for row in range(min_row, max_row + 1) if grid[row][min_col] == 0)), 0 if min_col > 0 else -1, 'L'), (sum((1 for row in range(min_row, max_row + 1) if grid[row][max_col] == 0)), 0 if max_col < len(grid[0]) - 1 else -1, 'R')]
    return max(calculate_gaps)[2]

def transpose(grid):
    return [list(row) for row in zip(*grid)]

def fill_right(grid, bounding_box, filled_grid):
    min_row, min_col, max_row, max_col = bounding_box
    height, width = (len(grid), len(grid[0]))
    for row in range(min_row, max_row + 1):
        if grid[row][max_col] != 0:
            continue
        for col in range(max_col + 1, width):
            if filled_grid[row][col] != 0:
                break
            filled_grid[row][col] = 4
    descending_row, ascending_col = (min_row + 1, max_col + 1)
    while 0 <= descending_row < height and 0 <= ascending_col < width:
        if filled_grid[descending_row][ascending_col] == 0:
            filled_grid[descending_row][ascending_col] = 4
        descending_row -= 1
        ascending_col += 1
    ascending_row, ascending_col = (max_row - 1, max_col + 1)
    while 0 <= ascending_row < height and 0 <= ascending_col < width:
        if filled_grid[ascending_row][ascending_col] == 0:
            filled_grid[ascending_row][ascending_col] = 4
        ascending_row += 1
        ascending_col += 1

def fill_left(grid, bounding_box, filled_grid):
    min_row, min_col, max_row, max_col = bounding_box
    height, _ = (len(grid), len(grid[0]))
    for row in range(min_row, max_row + 1):
        if grid[row][min_col] != 0:
            continue
        for col in range(min_col - 1, -1, -1):
            if filled_grid[row][col] != 0:
                break
            filled_grid[row][col] = 4
    descending_row, descending_col = (min_row + 1, min_col - 1)
    while 0 <= descending_row < height and 0 <= descending_col:
        if filled_grid[descending_row][descending_col] == 0:
            filled_grid[descending_row][descending_col] = 4
        descending_row -= 1
        descending_col -= 1
    ascending_row, descending_col = (max_row - 1, min_col - 1)
    while 0 <= ascending_row < height and 0 <= descending_col:
        if filled_grid[ascending_row][descending_col] == 0:
            filled_grid[ascending_row][descending_col] = 4
        ascending_row += 1
        descending_col -= 1

def fill_grid(grid):
    height, width = (len(grid), len(grid[0]))
    initial_grid = [row[:] for row in grid]
    values_set = {value for row in grid for value in row if value != 0}
    if values_set == {4}:
        initial_grid = [[1 if value == 4 else value for value in row] for row in grid]
    bounding_box = get_bounding_box(initial_grid)
    filled_grid = [row[:] for row in grid]
    for row in range(bounding_box[0], bounding_box[2] + 1):
        for col in range(bounding_box[1], bounding_box[3] + 1):
            if filled_grid[row][col] == 0:
                filled_grid[row][col] = 4
    chosen_direction = choose_direction(initial_grid, bounding_box)
    if chosen_direction == 'R':
        fill_right(grid, bounding_box, filled_grid)
    else:
        fill_left(grid, bounding_box, filled_grid)
    return filled_grid

def p(grid):
    values_set = {value for row in grid for value in row if value != 0}
    if values_set == {4}:
        transformed_grid = [[1 if value == 4 else value for value in row] for row in grid]
    else:
        transformed_grid = grid
    bounding_box = get_bounding_box(transformed_grid)
    chosen_direction = choose_direction(transformed_grid, bounding_box)
    if chosen_direction in ('L', 'R'):
        return fill_grid(grid)
    transposed_grid = transpose(grid)
    solve_transposed_grid = fill_grid(transposed_grid)
    return transpose(solve_transposed_grid)
