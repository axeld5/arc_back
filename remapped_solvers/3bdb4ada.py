def find_connected_color_group(grid, start_row, start_col, num_rows, num_cols):
    color = grid[start_row][start_col]
    connected_cells = set()
    if all((grid[start_row + offset][start_col] == color for offset in range(3))):
        end_col = start_col
        while end_col < num_cols and all((grid[start_row + offset][end_col] == color for offset in range(3))):
            for offset in range(3):
                connected_cells.add((start_row + offset, end_col))
            end_col += 1
    return connected_cells

def mark_inner_cells_zero(cleaned_grid, start_row, start_col, end_col):
    for col in range(start_col, end_col):
        if (col - start_col) % 2 == 1:
            cleaned_grid[start_row][col] = 0

def p(grid, value_range=range, get_length=len):
    cleaned_grid = [row[:] for row in grid]
    all_visited = set()
    num_rows = get_length(grid)
    num_cols = get_length(grid[0])
    for row in value_range(num_rows - 2):
        for col in value_range(num_cols):
            if grid[row][col] and (row, col) not in all_visited:
                connected_group = find_connected_color_group(grid, row, col, num_rows, num_cols)
                all_visited.update(connected_group)
                if connected_group:
                    min_col = min(connected_group, key=lambda x: x[1])[1]
                    max_col = max(connected_group, key=lambda x: x[1])[1] + 1
                    mark_inner_cells_zero(cleaned_grid, row + 1, min_col, max_col)
    return cleaned_grid
