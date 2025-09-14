def p(grid):

    def copy_non_negative(source, target, condition_index, target_index):
        if source[condition_index] > 0:
            target[target_index] = source[condition_index]
    for row_index in range(4):
        for col_offset in range(4):
            copy_non_negative(grid[row_index], grid[row_index], col_offset + 5, col_offset + 10)
            copy_non_negative(grid[row_index], grid[row_index], col_offset, col_offset + 10)
    return [row[10:] for row in grid]
