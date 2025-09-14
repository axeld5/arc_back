def solve(j):

    def is_within_grid(x, y, size):
        return 0 <= x < size and 0 <= y < size

    def mark_reachable_area(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if is_within_grid(cx, cy, grid_size) and (not visited[cx][cy]) and (j[cx][cy] == 0):
                visited[cx][cy] = 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))
    grid_size = len(j)
    visited = [[0] * grid_size for _ in range(grid_size)]
    for index in range(grid_size):
        mark_reachable_area(index, 0)
        mark_reachable_area(index, grid_size - 1)
        mark_reachable_area(0, index)
        mark_reachable_area(grid_size - 1, index)
    transformed_grid = [[3 if j[x][y] == 0 and (not visited[x][y]) else j[x][y] for y in range(grid_size)] for x in range(grid_size)]
    return [[3 if cell == 3 else 0 for cell in row] for row in transformed_grid]

def p(j):
    return solve(j)
