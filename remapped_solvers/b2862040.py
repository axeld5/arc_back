def p(grid, A=range):
    num_rows, num_cols = (len(grid), len(grid[0]))

    def neighbors(row, col):
        if row > 0:
            yield (row - 1, col)
        if row + 1 < num_rows:
            yield (row + 1, col)
        if col > 0:
            yield (row, col - 1)
        if col + 1 < num_cols:
            yield (row, col + 1)

    def find_components(target_value):
        seen = set()
        components = []
        for r in A(num_rows):
            for c in A(num_cols):
                if (r, c) in seen or grid[r][c] != target_value:
                    continue
                queue = [(r, c)]
                seen.add((r, c))
                connected_component = set()
                while queue:
                    i, j = queue.pop()
                    connected_component.add((i, j))
                    for ni, nj in neighbors(i, j):
                        if (ni, nj) not in seen and grid[ni][nj] == target_value:
                            seen.add((ni, nj))
                            queue.append((ni, nj))
                components.append(connected_component)
        return components

    def is_touching_boundary(component):
        return any((row == 0 or row == num_rows - 1 or col == 0 or (col == num_cols - 1) for row, col in component))
    components_of_9 = find_components(9)
    components_of_1 = find_components(1)
    internal_9s = set()
    for component in components_of_9:
        if not is_touching_boundary(component):
            internal_9s |= component
    to_update = set()
    for component in components_of_1:
        if any((any(((nr, nc) in internal_9s for nr, nc in neighbors(ir, ic))) for ir, ic in component)):
            to_update |= component
    for row, col in to_update:
        grid[row][col] = 8
    return grid
