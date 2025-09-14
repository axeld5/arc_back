def solve(grid, range_func=range):
    filled_grid = [row[:] for row in grid]
    for i in range_func(9):
        for j in range_func(9):
            block = [grid[i][j], grid[i][j + 1], grid[i + 1][j], grid[i + 1][j + 1]]
            non_zero_elements = [element for element in block if element]
            if len(non_zero_elements) == 3 and non_zero_elements.count(non_zero_elements[0]) == 3:
                zero_index = block.index(0)
                direction = get_fill_direction(zero_index)
                fill_value(filled_grid, i, j, zero_index, non_zero_elements[0], direction)
    return filled_grid

def get_fill_direction(zero_index):
    u_direction = -1 if zero_index < 2 else 1
    v_direction = -1 if zero_index % 2 == 0 else 1
    return (u_direction, v_direction)

def fill_value(grid, start_row, start_col, zero_index, value, direction):
    row_direction, col_direction = direction
    row = start_row + zero_index // 2 + row_direction
    col = start_col + zero_index % 2 + col_direction
    while 0 <= row < 10 and 0 <= col < 10:
        grid[row][col] = value
        row += row_direction
        col += col_direction
p = solve
