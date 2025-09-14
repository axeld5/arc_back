def fill_area(grid, row, col, label):
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
        return
    if grid[row][col]:
        return
    grid[row][col] = label
    for dr, dc in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        fill_area(grid, row + dr, col + dc, label)

def p(grid):
    size = 10
    fill_area(grid, 0, 0, 1)
    for index in range(4):
        row_offset = size // 2 - 1 + index % 2
        col_offset = size // 2 - 1 + index // 2
        fill_area(grid, row_offset, col_offset, 2)
    fill_area(grid, size - 1, size - 1, 3)
    return grid
