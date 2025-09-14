def p(grid):
    blocks = []
    for block_row in range(len(grid) // 3):
        for block_col in range(len(grid[0]) // 3):
            block = []
            for row in range(3):
                block_row_data = []
                for col in range(3):
                    block_row_data.append(grid[block_row * 3 + row][block_col * 3 + col])
                block.append(block_row_data)
            blocks.append(block)
    for block in blocks:
        pattern = tuple((tuple((cell == 0 for cell in row)) for row in block))
        pattern_count = 0
        for other_block in blocks:
            other_pattern = tuple((tuple((cell == 0 for cell in row)) for row in other_block))
            if pattern == other_pattern:
                pattern_count += 1
        if pattern_count == 1:
            return block
