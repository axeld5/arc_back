from collections import Counter

def get_most_common_element(flat_list):
    return Counter(flat_list).most_common(1)[0][0]

def get_non_empty_coordinates(grid, most_common_element):
    non_empty_coords = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value and value != most_common_element:
                non_empty_coords.append((i, j))
    return non_empty_coords

def generate_unique_sorted_coordinates(coordinate_pair_list):
    unique_rows = sorted({i for i, _ in coordinate_pair_list})
    unique_cols = sorted({j for _, j in coordinate_pair_list})
    return (unique_rows, unique_cols)

def generate_subgrid(grid, unique_rows, unique_cols, most_common_element):
    output_grid = []
    for i in range(len(unique_rows) - 1):
        row = []
        for j in range(len(unique_cols) - 1):
            value1 = grid[unique_rows[i]][unique_cols[j]]
            value2 = grid[unique_rows[i]][unique_cols[j + 1]]
            value3 = grid[unique_rows[i + 1]][unique_cols[j]]
            value4 = grid[unique_rows[i + 1]][unique_cols[j + 1]]
            if value1 == value2 == value3 == value4 and value1 not in (0, most_common_element):
                row.append(value1)
            else:
                row.append(0)
        output_grid.append(row)
    return output_grid

def p(grid):
    flattened_values = [value for row in grid for value in row if value]
    most_common_element = get_most_common_element(flattened_values)
    non_empty_coords = get_non_empty_coordinates(grid, most_common_element)
    unique_rows, unique_cols = generate_unique_sorted_coordinates(non_empty_coords)
    return generate_subgrid(grid, unique_rows, unique_cols, most_common_element)
