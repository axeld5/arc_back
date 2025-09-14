def p(grid):
    target_value = 0
    result = []
    for row in range(11):
        result_row = []
        for col in range(11):
            if (row + 1) % 4 == 0 or (col + 1) % 4 == 0:
                result_row.append(5)
            else:
                result_row.append(0)
        result.append(result_row)
    region_counts = {}
    for region_row in range(3):
        for region_col in range(3):
            region_counts[f'{region_row}{region_col}'] = 0
    for row, grid_row in enumerate(grid):
        for col, cell_value in enumerate(grid_row):
            if cell_value > 0 and cell_value != 5:
                target_value = int(cell_value)
                region_key = f'{row // 4}{col // 4}'
                region_counts[region_key] += 1
    max_count = max(region_counts.values())
    for row, result_row in enumerate(result):
        for col, cell_value in enumerate(result_row):
            if cell_value == 0:
                region_key = f'{row // 4}{col // 4}'
                if region_counts[region_key] == max_count:
                    result[row][col] = target_value
    return result
