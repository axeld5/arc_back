def p(grid, range_function=range):

    def find_blocks_to_modify(grid):
        num_rows, num_cols = (len(grid), len(grid[0]))
        blocks = []
        for row in range_function(num_rows - 1):
            for col in range_function(num_cols - 1):
                block = (grid[row][col], grid[row][col + 1], grid[row + 1][col], grid[row + 1][col + 1])
                if all(block):
                    blocks.append((row, col, len(set(block))))
        return blocks

    def modify_grid(grid, blocks):
        for start_row, start_col, unique_count in blocks:
            for offset in range_function(unique_count):
                row_to_modify = start_row + 2 + offset
                if row_to_modify < len(grid):
                    grid[row_to_modify][start_col] = 3
                    grid[row_to_modify][start_col + 1] = 3
    blocks_to_modify = find_blocks_to_modify(grid)
    modify_grid(grid, blocks_to_modify)
    return grid
