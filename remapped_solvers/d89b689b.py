def p(grid, iterator=enumerate):

    def construct_pairs(eligible_elements, excluded_indices):
        return [[row_index, col_index] for row_index, row in iterator(grid) for col_index, value in iterator(row) if value in eligible_elements and value not in excluded_indices]
    eligible_range, exclusion = (range(10), [0, 8])
    (row_e, col_k), (row_w, col_l), (row_j, col_a), (row_c, col_e) = construct_pairs(eligible_range, exclusion)
    key_row, key_col = construct_pairs([8], [])[0]
    grid[key_row][key_col:key_col + 2] = [grid[row_e][col_k], grid[row_w][col_l]][::(1, -1)[col_k > col_l]]
    grid[key_row + 1][key_col:key_col + 2] = [grid[row_j][col_a], grid[row_c][col_e]][::(1, -1)[col_a > col_e]]
    for row, col in ((row_e, col_k), (row_w, col_l), (row_j, col_a), (row_c, col_e)):
        grid[row][col] = 0
    return grid
