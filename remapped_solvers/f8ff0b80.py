from collections import Counter
from typing import List

def count_elements_in_grid(grid: List[List[int]]) -> Counter:
    flattened_elements = (element for row in grid for element in row)
    element_counter = Counter(flattened_elements)
    return element_counter

def get_most_common_elements(counter: Counter, n: int) -> List[int]:
    most_common_elements = counter.most_common(n + 1)
    return [element for element, count in most_common_elements[1:]]

def solve(grid: List[List[int]]) -> List[List[int]]:
    element_counter = count_elements_in_grid(grid)
    most_common_elements = get_most_common_elements(element_counter, 3)
    return [[element] for element in most_common_elements]
p = solve
