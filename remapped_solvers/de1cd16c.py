def p(grid, sort=sorted, length=len):
    height, width = (length(grid), length(grid[0]))
    frequency = {}
    for row in range(height):
        for value in grid[row]:
            frequency[value] = frequency.get(value, 0) + 1
    least_frequent_value = min(frequency, key=frequency.get)

    def most_common_in_row(row):
        count = {}
        for col in range(width):
            if grid[row][col] != least_frequent_value:
                count.update({grid[row][col]: count.get(grid[row][col], 0) + 1})
        sorted_items = sort(count.items(), key=lambda x: x[1], reverse=True)
        if length(sorted_items) > 1:
            return sort([sorted_items[0][0], sorted_items[1][0]])
        else:
            return sort([sorted_items[0][0]])

    def most_common_in_column(col):
        count = {}
        for row in range(height):
            if grid[row][col] != least_frequent_value:
                count.update({grid[row][col]: count.get(grid[row][col], 0) + 1})
        sorted_items = sort(count.items(), key=lambda x: x[1], reverse=True)
        if length(sorted_items) > 1:
            return sort([sorted_items[0][0], sorted_items[1][0]])
        else:
            return sort([sorted_items[0][0]])
    row_common_values = [most_common_in_row(r) for r in range(height)]
    col_common_values = [most_common_in_column(c) for c in range(width)]

    def find_split_index(common_values, max_index):
        return next((i for i in range(max_index) if common_values[i] and common_values[i + 1] and (common_values[i] != common_values[i + 1])), max_index // 2 - 1)
    X = find_split_index(row_common_values, height - 1)
    Y = find_split_index(col_common_values, width - 1)
    quadrants = [(0, X, 0, Y), (0, X, Y + 1, width - 1), (X + 1, height - 1, 0, Y), (X + 1, height - 1, Y + 1, width - 1)]
    max_placeholder_count = -1
    selected_quadrant = 0
    values_for_quadrants = [0] * 4
    for quadrant_index, (x_start, x_end, y_start, y_end) in enumerate(quadrants):
        placeholder_count = 0
        value_count = {}
        for r in range(x_start, x_end + 1):
            for c in range(y_start, y_end + 1):
                value = grid[r][c]
                if value == least_frequent_value:
                    placeholder_count += 1
                else:
                    value_count[value] = value_count.get(value, 0) + 1
        values_for_quadrants[quadrant_index] = max(value_count, key=value_count.get) if value_count else 0
        if placeholder_count > max_placeholder_count:
            max_placeholder_count = placeholder_count
            selected_quadrant = quadrant_index
    return [[values_for_quadrants[selected_quadrant]]]
