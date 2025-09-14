from collections import Counter, defaultdict

def p(grid):
    height, width = (len(grid), len(grid[0]))

    def is_within_bounds(row, col):
        return 0 <= row < height and 0 <= col < width
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    flattened_grid = [value for row in grid for value in row]
    most_common_elements = [element for element, _ in Counter(flattened_grid).most_common(2)]
    common_elements_set = set(most_common_elements)
    result_grid = [row[:] for row in grid]
    candidates = []
    for row in range(height):
        for col in range(width):
            cell_value = grid[row][col]
            if cell_value in common_elements_set:
                continue
            if any((is_within_bounds(row + dr, col + dc) and grid[row + dr][col + dc] == cell_value for dr, dc in neighbors)):
                continue
            neighbor_values = [grid[row + dr][col + dc] for dr, dc in neighbors if is_within_bounds(row + dr, col + dc)]
            if not neighbor_values:
                continue
            neighbor_common_counts = Counter((value for value in neighbor_values if value in common_elements_set))
            if not neighbor_common_counts:
                continue
            chosen_value = max(neighbor_common_counts, key=neighbor_common_counts.get)
            candidates.append((chosen_value, row, col, cell_value, neighbor_common_counts[chosen_value]))
    element_replacement_counts = defaultdict(Counter)
    element_weight_sums = defaultdict(lambda: defaultdict(int))
    for chosen_value, _, _, cell_value, count in candidates:
        element_replacement_counts[chosen_value][cell_value] += 1
        element_weight_sums[chosen_value][cell_value] += count
    first_common, second_common = most_common_elements
    first_candidates = list(element_replacement_counts[first_common].keys())
    second_candidates = list(element_replacement_counts[second_common].keys())
    best_score = None
    best_pair = None
    for first_candidate in first_candidates:
        for second_candidate in second_candidates:
            if first_candidate == second_candidate:
                continue
            score = (element_replacement_counts[first_common][first_candidate] + element_replacement_counts[second_common][second_candidate], element_weight_sums[first_common][first_candidate] + element_weight_sums[second_common][second_candidate])
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (first_candidate, second_candidate)
    first_candidate, second_candidate = best_pair
    replacements = {first_common: first_candidate, second_common: second_candidate}
    diagonal_neighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for chosen_value, row, col, cell_value, _ in candidates:
        for dr, dc in diagonal_neighbors:
            current_row, current_col = (row, col)
            while True:
                current_row += dr
                current_col += dc
                if not is_within_bounds(current_row, current_col):
                    break
                if grid[current_row][current_col] in replacements:
                    result_grid[current_row][current_col] = replacements[grid[current_row][current_col]]
    return result_grid
