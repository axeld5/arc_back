def find_bounding_box(coordinates):
    min_x = min((coord[1] for coord in coordinates))
    max_x = max((coord[1] for coord in coordinates))
    min_y = min((coord[0] for coord in coordinates))
    max_y = max((coord[0] for coord in coordinates))
    max_x = max_x - (max_x - min_x) // 2
    max_y = max_y - (max_y - min_y) // 2
    return (min_x, max_x, min_y, max_y)

def extract_coordinates_with_positive_values(grid):
    coordinates = []
    num_rows = len(grid)
    num_cols = len(grid[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] > 0:
                coordinates.append((row, col))
    return coordinates

def crop_grid(grid, min_col, max_col, min_row, max_row):
    cropped_grid = grid[min_row:max_row]
    cropped_grid = [row[min_col:max_col] for row in cropped_grid]
    return cropped_grid

def p(grid):
    positive_coordinates = extract_coordinates_with_positive_values(grid)
    min_x, max_x, min_y, max_y = find_bounding_box(positive_coordinates)
    cropped_grid = crop_grid(grid, min_x, max_x, min_y, max_y)
    return cropped_grid
