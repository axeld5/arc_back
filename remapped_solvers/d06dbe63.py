def p(grid):
    size = len(grid)
    start_row, start_col = next(((row, col) for row in range(size) for col in range(size) if grid[row][col]))
    for row_dir, col_dir in [(-1, 1), (1, -1)]:
        row, col = (start_row, start_col)
        while True:
            for _ in range(2):
                row += row_dir
                if 0 <= row < size:
                    grid[row][col] = 5
                else:
                    break
            else:
                for _ in range(2):
                    col += col_dir
                    if 0 <= col < size:
                        grid[row][col] = 5
                    else:
                        break
                else:
                    continue
            break
    return grid
