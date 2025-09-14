def mirror_half(grid_half):
    return [row + row[::-1] for row in grid_half]

def create_full_symmetric_grid(grid):
    mirrored_half = mirror_half(grid)
    return mirrored_half + mirrored_half[::-1]

def p(grid):
    return create_full_symmetric_grid(grid)
