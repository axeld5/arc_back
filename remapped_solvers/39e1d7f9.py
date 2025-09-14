def p(input_grid):

    def row_unification(row_index):
        first_value = input_grid[row_index][0]
        if all((input_grid[row_index][col] == first_value for col in range(grid_width))):
            return first_value
        return None

    def column_unification(col_index):
        first_value = input_grid[0][col_index]
        if all((input_grid[row][col_index] == first_value for row in range(grid_height))):
            return first_value
        return None

    def most_common_value(values, ignore_value=None):
        frequency = {}
        highest_count, most_common = (-1, 0)
        for value in values:
            if value == ignore_value:
                continue
            frequency[value] = frequency.get(value, 0) + 1
            if frequency[value] > highest_count:
                highest_count, most_common = (frequency[value], value)
        return most_common

    def compute_segments(indices, max_value):
        boundaries = [-1] + indices + [max_value]
        return [range(boundaries[i] + 1, boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    grid_height, grid_width = (len(input_grid), len(input_grid[0]))
    unification_counts = {}
    for row in range(grid_height):
        unified_value = row_unification(row)
        if unified_value is not None:
            unification_counts[unified_value] = unification_counts.get(unified_value, 0) + 1
    for col in range(grid_width):
        unified_value = column_unification(col)
        if unified_value is not None:
            unification_counts[unified_value] = unification_counts.get(unified_value, 0) + 1
    most_unified_value = max(unification_counts, key=unification_counts.get)
    unified_rows = [row for row in range(grid_height) if row_unification(row) == most_unified_value]
    unified_columns = [col for col in range(grid_width) if column_unification(col) == most_unified_value]
    row_segments = compute_segments(unified_rows, grid_height)
    column_segments = compute_segments(unified_columns, grid_width)
    segment_rows, segment_cols = (len(row_segments), len(column_segments))
    largest_segment = [[0 for _ in range(segment_cols)] for _ in range(segment_rows)]
    for i in range(segment_rows):
        row_range = row_segments[i]
        for j in range(segment_cols):
            col_range = column_segments[j]
            segment_values = [input_grid[row][col] for row in row_range for col in col_range]
            largest_segment[i][j] = most_common_value(segment_values, most_unified_value)
    directions_straight = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    all_directions = directions_straight + directions_diagonal

    def in_bounds(i, j):
        return 0 <= i < segment_rows and 0 <= j < segment_cols
    max_adjacents, max_position, adjacency_map = ((-1, -1), None, {})
    for i in range(segment_rows):
        for j in range(segment_cols):
            current_value = largest_segment[i][j]
            if current_value == 0:
                continue
            adjacency_counts = {}
            diagonal_touch = 0
            for di, dj in all_directions:
                adj_i, adj_j = (i + di, j + dj)
                if not in_bounds(adj_i, adj_j):
                    continue
                adjacent_value = largest_segment[adj_i][adj_j]
                if adjacent_value != 0 and adjacent_value != current_value:
                    adjacency_counts[di, dj] = adjacent_value
                    if (di, dj) in directions_diagonal:
                        diagonal_touch = 1
            adjacency_data = (len(adjacency_counts), diagonal_touch)
            if adjacency_counts and adjacency_data > max_adjacents:
                max_adjacents, max_position, adjacency_map = (adjacency_data, (i, j), adjacency_counts)
    selected_value = largest_segment[max_position[0]][max_position[1]]
    final_grid = [[cell for cell in row] for row in input_grid]
    output_segment_grid = [[cell for cell in row] for row in largest_segment]
    for i in range(segment_rows):
        for j in range(segment_cols):
            if largest_segment[i][j] == selected_value:
                for (di, dj), adjacent_value in adjacency_map.items():
                    adj_i, adj_j = (i + di, j + dj)
                    if in_bounds(adj_i, adj_j):
                        output_segment_grid[adj_i][adj_j] = adjacent_value
    for i in range(segment_rows):
        row_range = row_segments[i]
        for j in range(segment_cols):
            col_range = column_segments[j]
            most_common = output_segment_grid[i][j]
            for row in row_range:
                for col in col_range:
                    final_grid[row][col] = most_common
    for row in unified_rows:
        final_grid[row] = [most_unified_value] * grid_width
    for col in unified_columns:
        for row in range(grid_height):
            final_grid[row][col] = most_unified_value
    return final_grid
