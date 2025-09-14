from collections import Counter

def solve_frequency_to_grid(jumble):

    def count_frequencies(jumble):
        flattened_list = [element for row in jumble for element in row]
        return Counter(flattened_list).most_common(9)

    def construct_grid(most_common, max_count, unique_count):
        grid = [[0 for _ in range(unique_count)] for _ in range(max_count)]
        for column_index in range(unique_count):
            number, frequency = most_common[column_index]
            for row_index in range(frequency):
                grid[row_index][column_index] = number
        return grid
    most_common = count_frequencies(jumble)
    max_frequency = most_common[0][1]
    unique_count = len(most_common)
    return construct_grid(most_common, max_frequency, unique_count)
p = solve_frequency_to_grid
