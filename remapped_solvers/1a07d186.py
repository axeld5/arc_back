def p(grid):
    if has_complete_column(grid):
        return transform_grid(grid)
    else:
        return transpose_grid(transform_grid(transpose_grid(grid)))

def has_complete_column(grid):
    for column_index in range(len(grid[0])):
        if all((grid[row][column_index] == grid[0][column_index] != 0 for row in range(len(grid)))):
            return True
    return False

def transpose_grid(grid):
    return [list(row) for row in zip(*grid)]

def transform_grid(grid):
    height, width = (len(grid), len(grid[0]))
    columns_to_preserve = [index for index in range(width) if all((grid[row][index] == grid[0][index] != 0 for row in range(height)))]
    transformed_grid = [[0] * width for _ in range(height)]
    for col_index in columns_to_preserve:
        for row_index in range(height):
            transformed_grid[row_index][col_index] = grid[row_index][col_index]
    for row_index in range(height):
        for col_index in range(width):
            value = grid[row_index][col_index]
            if value == 0 or col_index in columns_to_preserve:
                continue
            candidates = [b for b in columns_to_preserve if grid[0][b] == value]
            if candidates:
                best_match_col = min(candidates, key=lambda q: abs(q - col_index))
                adjusted_col = best_match_col - 1 if col_index < best_match_col else best_match_col + 1
                transformed_grid[row_index][adjusted_col] = value
    return transformed_grid
