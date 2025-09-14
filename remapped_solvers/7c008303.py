def extract_subgrid(grid, start_row, start_col):
    return [grid[start_row][start_col:start_col + 2], grid[start_row + 1][start_col:start_col + 2]]

def get_special_position_indices(grid):
    special_row = next((row for row in range(9) if all((value == 8 for value in grid[row]))))
    special_col = next((col for col in range(9) if all((grid[row][col] == 8 for row in range(9)))))
    return (special_row, special_col)

def find_valid_subgrid_corner(grid):
    corners = {'T': (0, 0), 'R': (0, 7), 'L': (7, 0), 'B': (7, 7)}
    for corner_name, (start_row, start_col) in corners.items():
        subgrid = extract_subgrid(grid, start_row, start_col)
        flattened_subgrid = subgrid[0] + subgrid[1]
        if all((value != 0 and value != 8 for value in flattened_subgrid)) and 3 not in flattened_subgrid:
            return (corner_name, subgrid)
    return (None, None)

def p(grid, range_func=range):
    special_row, special_col = get_special_position_indices(grid)
    corner_name, valid_subgrid = find_valid_subgrid_corner(grid)
    if corner_name is None or valid_subgrid is None:
        return None
    if corner_name == 'T':
        start_row, start_col = (special_row + 1, special_col + 1)
    elif corner_name == 'R':
        start_row, start_col = (special_row + 1, 0)
    elif corner_name == 'L':
        start_row, start_col = (0, special_col + 1)
    elif corner_name == 'B':
        start_row, start_col = (0, 0)
    T, t, p, L = (valid_subgrid[0][0], valid_subgrid[0][1], valid_subgrid[1][0], valid_subgrid[1][1])
    output_size = 6
    output_grid = [[0] * output_size for _ in range_func(output_size)]
    for u in range_func(start_row, min(start_row + output_size, 9)):
        for v in range_func(start_col, min(start_col + output_size, 9)):
            if grid[u][v] == 3:
                m, w = (u - start_row, v - start_col)
                middle_point = output_size // 3
                if m <= middle_point and w <= middle_point:
                    output_grid[m][w] = T
                elif m <= middle_point and w >= middle_point:
                    output_grid[m][w] = t
                elif m >= middle_point and w <= middle_point:
                    output_grid[m][w] = p
                elif m >= middle_point and w >= middle_point:
                    output_grid[m][w] = L
    return output_grid
