def p(grid, range_=range):
    height, width = (len(grid), len(grid[0]))

    def count_frequencies(grid):
        frequencies = {}
        for row in grid:
            for value in row:
                if value:
                    frequencies[value] = frequencies.get(value, 0) + 1
        return frequencies
    frequencies = count_frequencies(grid)
    most_frequent_value = max(frequencies, key=frequencies.get) if frequencies else 0
    pattern_to_index = {11: 0, 23: 1, 38: 2, 89: 3, 0: 4, 308: 5, 200: 6, 464: 7, 416: 8}
    output_grid = [[most_frequent_value] * 9 for _ in range_(9)]
    for x in range_(height - 2):
        for y in range_(width - 2):
            subgrid = [grid[x + i][y + j] for i in range(3) for j in range(3)]
            if all(subgrid):
                mask = 0
                first_non_frequent_value = None
                for index, value in enumerate(subgrid):
                    if value != most_frequent_value:
                        mask |= 1 << index
                        if first_non_frequent_value is None:
                            first_non_frequent_value = value
                if mask in pattern_to_index:
                    row_offset = 3 * (pattern_to_index[mask] // 3)
                    col_offset = 3 * (pattern_to_index[mask] % 3)
                    for index in range(9):
                        if mask & 1 << index:
                            output_grid[row_offset + index // 3][col_offset + index % 3] = first_non_frequent_value
    return output_grid
