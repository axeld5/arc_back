from collections import Counter

def find_most_common_with_frequency(elements, frequency):
    element_counts = Counter(elements).most_common()
    for element, count in element_counts:
        if count == frequency:
            return element
    return None

def find_bounding_box(coordinates):
    min_row = min((coord[0] for coord in coordinates))
    max_row = max((coord[0] for coord in coordinates))
    min_col = min((coord[1] for coord in coordinates))
    max_col = max((coord[1] for coord in coordinates))
    return (min_row, max_row, min_col, max_col)

def p(grid):
    all_elements = [element for row in grid for element in row]
    target_value = find_most_common_with_frequency(all_elements, 4)
    if target_value is None:
        return grid
    target_coords = [(row_index, col_index) for row_index, row in enumerate(grid) for col_index, value in enumerate(row) if value == target_value]
    min_row, max_row, min_col, max_col = find_bounding_box(target_coords)
    cropped_grid = []
    for row in grid[min_row + 1:max_row]:
        new_row = [value if value <= 0 else target_value for value in row[min_col + 1:max_col]]
        cropped_grid.append(new_row)
    return cropped_grid
