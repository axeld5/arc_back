def p(I, range=range):
    ROAD, CITY, HIGHWAY = (2, 8, 4)
    height, width = (len(I), len(I[0]))
    output_grid = I[:]

    def find_longest_road(column):
        best_start, longest_length = (None, 0)
        row = 0
        while row < height:
            if I[row][column] == ROAD:
                start = row
                while row < height and I[row][column] == ROAD:
                    row += 1
                length = row - start
                if length > longest_length:
                    best_start, longest_length = (start, length)
            else:
                row += 1
        return (best_start, longest_length)
    left_start, left_length = find_longest_road(0)
    right_start, right_length = find_longest_road(width - 1)
    minimal_length = min(left_length, right_length)

    def has_city_in_road_segment(start):
        for row in range(start, start + minimal_length):
            if CITY in I[row]:
                return True
        return False
    if has_city_in_road_segment(left_start):
        active_start, other_start = (left_start, right_start)
        primary_col, secondary_col = (0, width - 1)
    else:
        active_start, other_start = (right_start, left_start)
        primary_col, secondary_col = (width - 1, 0)
    for k in range(minimal_length):
        active_row = active_start + k
        city_index = None
        if primary_col == 0:
            for col in range(1, width - 1):
                if I[active_row][col] == CITY:
                    city_index = col
                    break
        else:
            for col in range(width - 2, 0, -1):
                if I[active_row][col] == CITY:
                    city_index = col
                    break
        if city_index is None:
            continue
        output_grid[active_row][city_index] = HIGHWAY
        if primary_col == 0:
            output_grid[active_row][0] = ROAD
            for col in range(1, city_index):
                output_grid[active_row][col] = CITY
        else:
            output_grid[active_row][width - 1] = ROAD
            for col in range(city_index + 1, width - 1):
                output_grid[active_row][col] = CITY
        other_row = other_start + k
        if secondary_col == width - 1:
            for col in range(width - 1):
                output_grid[other_row][col] = CITY
            output_grid[other_row][width - 1] = ROAD
        else:
            output_grid[other_row][0] = ROAD
            for col in range(1, width):
                output_grid[other_row][col] = CITY
    return output_grid
