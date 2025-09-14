def p(grid, default_layer1=[2] * 3, default_layer2=[0] * 3):
    layered_structures = [[default_layer1, [0, 2, 0], default_layer2], [default_layer1, default_layer2, default_layer2], [[2, 2, 0], default_layer2, default_layer2], [[2, 0, 0], default_layer2, default_layer2]]
    count_of_ones = sum((row.count(1) for row in grid))
    return layered_structures[4 - count_of_ones]
