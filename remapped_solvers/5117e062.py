def p(grid):
    for row in range(1, len(grid) - 1):
        for col in range(1, len(grid[0]) - 1):
            if grid[row][col] == 8:
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if (dr != 0 or dc != 0) and grid[row + dr][col + dc]:
                            neighbors.append(grid[row + dr][col + dc])
                most_frequent = max(set(neighbors), key=neighbors.count)
                pattern = [[grid[row + dr][col + dc] for dc in [-1, 0, 1]] for dr in [-1, 0, 1]]
                pattern[1][1] = most_frequent
                return pattern
