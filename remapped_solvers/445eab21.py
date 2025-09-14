def p(grid):

    def flatten_grid(grid):
        return sum(grid, [])

    def most_common_element(elements):
        return max(elements, key=elements.count)

    def custom_max_element(elements):
        return max(elements, key=lambda k: elements.count(k) if k else -1)
    flattened_grid = flatten_grid(grid)
    common_element = most_common_element(flattened_grid)
    selected_element = common_element or custom_max_element(flattened_grid)
    return [[selected_element, selected_element], [selected_element, selected_element]]
