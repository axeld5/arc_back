def find_largest_subrectangle_with_most_twos(grid, size=10):

    def get_subrectangle(grid, start_row, start_col, height, width):
        return [row[start_col:start_col + width] for row in grid[start_row:start_row + height]]
    max_twos = -1
    best_rectangle = None
    for row in range(size):
        for col in range(size):
            if grid[row][col] and (row == 0 or grid[row - 1][col] < 1) and (col == 0 or grid[row][col - 1] < 1):
                width = 1
                height = 1
                while col + width < size and grid[row][col + width]:
                    width += 1
                while row + height < size and grid[row + height][col]:
                    height += 1
                subrectangle = get_subrectangle(grid, row, col, height, width)
                twos_count = sum((row.count(2) for row in subrectangle))
                if twos_count > max_twos:
                    max_twos = twos_count
                    best_rectangle = subrectangle
    return best_rectangle
p = find_largest_subrectangle_with_most_twos
