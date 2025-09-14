def p(grid):
    result = []
    for row in range(9):
        result_row = []
        for col in range(9):
            grid_row = row // 3
            grid_col = col // 3
            pattern_row = row % 3
            pattern_col = col % 3
            value = grid[grid_row][grid_col] and grid[pattern_row][pattern_col]
            result_row.append(value)
        result.append(result_row)
    return result
