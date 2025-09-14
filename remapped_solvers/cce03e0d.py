def p(grid):
    result = []
    for row in range(9):
        output_row = []
        for col in range(9):
            element = grid[row % 3][col % 3]
            block_selector = grid[row // 3][col // 3]
            output_row.append(element if block_selector == 2 else 0)
        result.append(output_row)
    return result
