def calculate_edge_sum(grid_slice, aggregation_func=max):
    return aggregation_func(map(aggregation_func, grid_slice))

def solve(grid, aggregation_func=max, column_range=range(10)):
    first_half_max = calculate_edge_sum(grid[:5], aggregation_func)
    second_half_max = calculate_edge_sum(grid[-5:], aggregation_func)
    new_grid = []
    for row_index in column_range:
        new_row = []
        for col_index in column_range:
            if row_index in (0, 2, 7, 9) or col_index % 9 < 1:
                if row_index < 5:
                    new_row.append(first_half_max)
                else:
                    new_row.append(second_half_max)
            else:
                new_row.append(0)
        new_grid.append(new_row)
    return new_grid
p = solve
