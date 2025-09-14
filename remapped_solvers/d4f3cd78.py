def p(grid, H=10, K=range):

    def find_fillable_area_positions():
        positions = []
        for row in K(H):
            for col in K(H):
                if grid[row][col] == 5:
                    positions.append((row, col))
        return positions

    def find_bounding_coordinates(positions):
        min_row = min((row for row, _ in positions))
        max_row = max((row for row, _ in positions))
        min_col = min((col for _, col in positions))
        max_col = max((col for _, col in positions))
        return (min_row, max_row, min_col, max_col)

    def mark_inner_area_as_filled(min_row, max_row, min_col, max_col):
        for row in K(min_row + 1, max_row):
            for col in K(min_col + 1, max_col):
                grid[row][col] = 8

    def identify_unfillable_edges(min_row, max_row, min_col, max_col):
        top_edge = [(min_row, col) for col in K(min_col, max_col + 1) if grid[min_row][col] != 5]
        bottom_edge = [(max_row, col) for col in K(min_col, max_col + 1) if grid[max_row][col] != 5]
        left_edge = [(row, min_col) for row in K(min_row, max_row + 1) if grid[row][min_col] != 5]
        right_edge = [(row, max_col) for row in K(min_row, max_row + 1) if grid[row][max_col] != 5]
        return {'bottom': bottom_edge, 'right': right_edge, 'left': left_edge, 'top': top_edge}

    def fill_based_on_longest_unfillable_edge(edges):
        longest_edge_key = max(edges, key=lambda x: len(edges[x]))
        start_row, start_col = edges[longest_edge_key][0]
        if longest_edge_key == 'bottom':
            for row in K(max_row, H):
                grid[row][start_col] = 8
        elif longest_edge_key == 'top':
            for row in K(min_row, -1, -1):
                grid[row][start_col] = 8
        elif longest_edge_key == 'left':
            for col in K(min_col, -1, -1):
                grid[start_row][col] = 8
        elif longest_edge_key == 'right':
            for col in K(max_col, H):
                grid[start_row][col] = 8
    positions_5 = find_fillable_area_positions()
    min_row, max_row, min_col, max_col = find_bounding_coordinates(positions_5)
    mark_inner_area_as_filled(min_row, max_row, min_col, max_col)
    unfillable_edges = identify_unfillable_edges(min_row, max_row, min_col, max_col)
    fill_based_on_longest_unfillable_edge(unfillable_edges)
    return grid
