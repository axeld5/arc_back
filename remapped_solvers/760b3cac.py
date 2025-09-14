def p(j):
    start_col = 6 if j[3][3] < 1 else 0
    for row in range(3):
        j[row][start_col:start_col + 3] = j[row][3:6][::-1]
    return j
