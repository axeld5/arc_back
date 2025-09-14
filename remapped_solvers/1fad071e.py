def p(grid):
    count = 0
    for i in range(8):
        for j in range(8):
            if all((grid[i + k][j + l] == 1 for k in range(2) for l in range(2))):
                count += 1
    return [[1 if i < count else 0 for i in range(5)]]
