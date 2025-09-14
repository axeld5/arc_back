def find_period(grid, is_row=True):
    size = len(grid) if is_row else len(grid[0])
    other_size = len(grid[0]) if is_row else len(grid)
    for period in range(1, other_size):
        if all((cell_matches(cell1, cell2) for index in range(size) for cell1, cell2 in zip(get_line(grid, index, is_row), get_line(grid, index + period, is_row)))):
            return period
    return other_size

def cell_matches(cell1, cell2):
    return cell1 == cell2 or cell1 * cell2 < 1

def get_line(grid, index, is_row=True):
    if is_row:
        return grid[index] if index < len(grid) else []
    else:
        return [row[index] for row in grid if index < len(row)]

def assemble_decorated_dict(grid, row_period, column_period):
    decorated_dict = {}
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if value:
                decorated_dict[row_index % row_period, col_index % column_period] = value
    return decorated_dict

def p(grid):
    if not grid or not grid[0]:
        return grid
    row_period = find_period(grid, is_row=True)
    column_period = find_period(grid, is_row=False)
    decorated_dict = assemble_decorated_dict(grid, row_period, column_period)
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            if not value:
                grid[row_index][col_index] = decorated_dict[row_index % row_period, col_index % column_period]
    return grid
