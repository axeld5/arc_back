def count_frequencies(grid):
    from collections import Counter
    flat_list = [element for row in grid for element in row if element]
    return dict(Counter(flat_list).most_common())

def sort_elements_by_frequency(element_count):
    return sorted(element_count, key=element_count.get, reverse=True)

def distribute_elements(sorted_elements, element_count, num_rows, num_cols):
    result_grid = [[0] * num_cols for _ in range(num_rows)]
    for index, element in enumerate(sorted_elements):
        result_grid[-1 - index][-element_count[element]:] = [element] * element_count[element]
    return result_grid

def p(grid):
    if not grid or not grid[0]:
        return []
    num_cols = len(grid[0])
    element_count = count_frequencies(grid)
    sorted_elements = sort_elements_by_frequency(element_count)
    num_rows = len(grid)
    return distribute_elements(sorted_elements, element_count, num_rows, num_cols)
