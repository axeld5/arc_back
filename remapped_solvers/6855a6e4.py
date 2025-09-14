def p(A, P=range, E=enumerate, S=sorted):
    R = 2
    G = 5
    K = P(15)
    row_count = [sum((cell == R for cell in A[row])) for row in K]
    col_count = [sum((A[row][col] == R for row in K)) for col in K]
    row_indices = [i for i, count in E(row_count) if count >= 3] or S(K, key=lambda i: -row_count[i])[:2]
    col_indices = [i for i, count in E(col_count) if count >= 3] or S(K, key=lambda i: -col_count[i])[:2]
    top_row, bottom_row = S(set(row_indices))[:2]
    left_col, right_col = S(set(col_indices))[:2]
    subgrid = [A[r][left_col:right_col + 1] for r in P(top_row, bottom_row + 1)]
    for i in P(bottom_row - top_row + 1):
        for j in P(right_col - left_col + 1):
            if subgrid[i][j] == G:
                subgrid[i][j] = 0
    height = bottom_row - top_row + 1
    width = right_col - left_col + 1
    mid_height = (height - 1) // 2
    mid_width = (width - 1) // 2
    if row_count[top_row] + row_count[bottom_row] >= col_count[left_col] + col_count[right_col]:
        for col in P(left_col + 1, right_col):
            for row in K:
                if A[row][col] == G:
                    spread_row = 2 * top_row - row if row < top_row else 2 * top_row + height + mid_height + 3 - row if row > bottom_row else -1
                    if top_row + 2 <= spread_row <= bottom_row - 2:
                        subgrid[spread_row - top_row][col - left_col] = G
    else:
        for row in P(top_row + 1, bottom_row):
            for col in K:
                if A[row][col] == G:
                    spread_col = 2 * left_col - col if col < left_col else 2 * left_col + width + mid_width + 3 - col if col > right_col else -1
                    if left_col + 1 <= spread_col <= right_col - 1:
                        subgrid[row - top_row][spread_col - left_col] = G
    output_grid = [[0] * 15 for _ in K]
    for r in P(top_row, bottom_row + 1):
        output_grid[r][left_col:right_col + 1] = subgrid[r - top_row]
    return output_grid
