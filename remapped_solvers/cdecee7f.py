def extract_column_non_zero_elements(grid):
    non_zero_elements = []
    for column in zip(*grid):
        for value in column:
            if value != 0:
                non_zero_elements.append(value)
                break
    return non_zero_elements

def fill_with_zeros(elements, size):
    elements.extend([0] * (size - len(elements)))
    return elements

def rearrange_in_snake_pattern(elements, dimension):
    grid = []
    for i in range(dimension):
        row = elements[i * dimension:(i + 1) * dimension]
        if i % 2 == 1:
            row.reverse()
        grid.append(row)
    return grid

def p(grid):
    SIZE = 3
    non_zero_elements = extract_column_non_zero_elements(grid)
    filled_elements = fill_with_zeros(non_zero_elements, SIZE * SIZE)
    return rearrange_in_snake_pattern(filled_elements, SIZE)
