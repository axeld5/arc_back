def p(j, R=range, A=enumerate, c=next):

    def can_be_edge_of_subgrid(row_idx, col_idx):
        return row_idx < 1 or j[row_idx - 1][col_idx] < 1 or (row_idx > 2 and j[row_idx - 1][col_idx] == 4 and (j[row_idx - 2][col_idx] > 0))

    def locate_special_values():
        positions = [divmod(idx, 13) for idx, val in A(sum(j, [])) if val == 4]
        (start_row, start_col), (end_row, end_col), (a_row, last_row), last_col = positions
        return (start_row, start_col, end_row, end_col, a_row, last_row, last_col)

    def locate_common_value():
        return c((u for row in zip(*j) if 4 not in row for u in row if u))

    def locate_edges(common_value):
        e_index = c((idx for idx, row in A(j) if any((u == common_value and can_be_edge_of_subgrid(idx, col) for col, u in A(row)))))
        k_index = c((idx for idx, col_values in A(zip(*j)) if any((u == common_value and can_be_edge_of_subgrid(row, idx) for row, u in A(col_values)))))
        return (e_index, k_index)
    start_row, start_col, end_row, end_col, a_row, last_row, last_col = locate_special_values()
    common_value = locate_common_value()
    e_index, k_index = locate_edges(common_value)
    for row_offset in R(a_row - start_row - 1):
        for col_offset in R(end_col - start_col - 1):
            swap_idx = [end_col - col_offset - 1, start_col + col_offset + 1][j[start_row + 1][start_col] == common_value]
            j[start_row + row_offset + 1][swap_idx], j[e_index + row_offset][k_index + col_offset] = (j[e_index + row_offset][k_index + col_offset], 0)
    return [row[start_col:end_col + 1] for row in j[start_row:a_row + 1]]
