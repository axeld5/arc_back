def find_vertical_triplet(grid, row_idx, col_idx):
    return [grid[row_idx + offset][col_idx - 1:col_idx + 2] for offset in (1, 2, 3)]

def locate_five(grid):
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == 5:
                return (i, j)

def solve(grid):
    row_idx, col_idx = locate_five(grid)
    return find_vertical_triplet(grid, row_idx, col_idx)
p = solve
