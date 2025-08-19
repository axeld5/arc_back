"""
Augmentation functions for ARC (Abstraction and Reasoning Corpus) problems.

This module provides various augmentation functions that can be applied to grid-based problems
to create variations for data augmentation purposes.
"""

import random
import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from copy import deepcopy


def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    """
    Rotate a grid 90 degrees clockwise.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Rotated grid
    """
    if not grid or not grid[0]:
        return grid
    
    rows, cols = len(grid), len(grid[0])
    rotated = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            rotated[j][rows - 1 - i] = grid[i][j]
    
    return rotated


def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    """
    Rotate a grid 180 degrees.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Rotated grid
    """
    if not grid:
        return grid
    
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: List[List[int]]) -> List[List[int]]:
    """
    Rotate a grid 270 degrees clockwise (or 90 degrees counter-clockwise).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Rotated grid
    """
    if not grid or not grid[0]:
        return grid
    
    rows, cols = len(grid), len(grid[0])
    rotated = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            rotated[cols - 1 - j][i] = grid[i][j]
    
    return rotated


def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """
    Flip a grid vertically (top-bottom).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Vertically flipped grid
    """
    return grid[::-1]


def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """
    Flip a grid horizontally (left-right).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Horizontally flipped grid
    """
    return [row[::-1] for row in grid]


def apply_color_permutation(grid: List[List[int]], color_map: Dict[int, int] = None) -> List[List[int]]:
    """
    Apply a color permutation to a grid. Colors are in range 0-9.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        color_map (Dict[int, int], optional): Mapping from old colors to new colors.
                                            If None, generates a random permutation.
        
    Returns:
        List[List[int]]: Grid with colors permuted
    """
    if color_map is None:
        # Create a random permutation of colors 0-9
        colors = list(range(10))
        shuffled_colors = colors.copy()
        random.shuffle(shuffled_colors)
        color_map = dict(zip(colors, shuffled_colors))
    
    return [[color_map.get(cell, cell) for cell in row] for row in grid]


def upscale_grid(grid: List[List[int]], target_size: Tuple[int, int] = None) -> List[List[int]]:
    """
    Upscale a grid by padding with zeros to reach a larger size.
    The original grid can be placed anywhere within the new larger grid.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        target_size (Tuple[int, int], optional): Target (height, width). If None, randomly chosen up to 30x30.
        
    Returns:
        List[List[int]]: Upscaled grid with zero padding
    """
    if not grid or not grid[0]:
        return grid
    
    current_height = len(grid)
    current_width = len(grid[0])
    
    if target_size is None:
        # Randomly choose target size between current size and 30x30
        max_height = min(30, current_height + random.randint(1, 30 - current_height))
        max_width = min(30, current_width + random.randint(1, 30 - current_width))
        target_height = random.randint(current_height, max_height)
        target_width = random.randint(current_width, max_width)
    else:
        target_height, target_width = target_size
    
    # Ensure target size is at least as large as current size
    target_height = max(target_height, current_height)
    target_width = max(target_width, current_width)
    
    # Randomly choose where to place the original grid within the new grid
    max_top_offset = target_height - current_height
    max_left_offset = target_width - current_width
    
    top_offset = random.randint(0, max_top_offset)
    left_offset = random.randint(0, max_left_offset)
    
    # Create the new grid filled with zeros
    upscaled = [[0] * target_width for _ in range(target_height)]
    
    # Place the original grid at the chosen position
    for i in range(current_height):
        for j in range(current_width):
            upscaled[top_offset + i][left_offset + j] = grid[i][j]
    
    return upscaled


def apply_augmentation_to_example(example: Dict[str, List[List[int]]], 
                                augmentation_func: Callable,
                                **kwargs) -> Dict[str, List[List[int]]]:
    """
    Apply an augmentation function to both input and output of an example.
    
    Args:
        example (Dict[str, List[List[int]]]): Example with 'input' and 'output' keys
        augmentation_func (Callable): Augmentation function to apply
        **kwargs: Additional arguments for the augmentation function
        
    Returns:
        Dict[str, List[List[int]]]: Augmented example
    """
    augmented_example = {}
    
    if 'input' in example:
        augmented_example['input'] = augmentation_func(example['input'], **kwargs)
    
    if 'output' in example:
        augmented_example['output'] = augmentation_func(example['output'], **kwargs)
    
    return augmented_example


def apply_augmentation_to_problem(problem: Dict[str, Any], 
                                augmentation_func: Callable,
                                **kwargs) -> Dict[str, Any]:
    """
    Apply an augmentation function to all examples in a problem.
    
    Args:
        problem (Dict[str, Any]): Problem dictionary with 'train', 'test', etc.
        augmentation_func (Callable): Augmentation function to apply
        **kwargs: Additional arguments for the augmentation function
        
    Returns:
        Dict[str, Any]: Problem with augmentation applied to all examples
    """
    augmented_problem = deepcopy(problem)
    
    # Apply to training examples
    if 'train' in augmented_problem:
        augmented_problem['train'] = [
            apply_augmentation_to_example(example, augmentation_func, **kwargs)
            for example in augmented_problem['train']
        ]
    
    # Apply to test examples
    if 'test' in augmented_problem:
        augmented_problem['test'] = [
            apply_augmentation_to_example(example, augmentation_func, **kwargs)
            for example in augmented_problem['test']
        ]
    
    # Apply to arc-gen examples if present
    if 'arc-gen' in augmented_problem:
        augmented_problem['arc-gen'] = [
            apply_augmentation_to_example(example, augmentation_func, **kwargs)
            for example in augmented_problem['arc-gen']
        ]
    
    return augmented_problem


def get_available_augmentations() -> Dict[str, Callable]:
    """
    Get a dictionary of all available augmentation functions.
    
    Returns:
        Dict[str, Callable]: Dictionary mapping augmentation names to functions
    """
    return {
        'rotate_90': rotate_90,
        'rotate_180': rotate_180,
        'rotate_270': rotate_270,
        'flip_vertical': flip_vertical,
        'flip_horizontal': flip_horizontal,
        'color_permutation': apply_color_permutation,
        'upscale': upscale_grid
    }


def apply_random_augmentations(problem: Dict[str, Any], 
                             num_augmentations: int = None,
                             seed: int = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply a random selection of 2-4 augmentations to a whole problem.
    
    Args:
        problem (Dict[str, Any]): Problem dictionary to augment
        num_augmentations (int, optional): Number of augmentations to apply (2-4).
                                         If None, randomly chosen.
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        Tuple[Dict[str, Any], List[str]]: Augmented problem and list of applied augmentations
    """
    if seed is not None:
        random.seed(seed)
    
    if num_augmentations is None:
        num_augmentations = random.randint(2, 4)
    else:
        num_augmentations = max(2, min(4, num_augmentations))
    
    available_augmentations = get_available_augmentations()
    augmentation_names = list(available_augmentations.keys())
    
    # Randomly select augmentations without replacement
    selected_augmentations = random.sample(augmentation_names, num_augmentations)
    
    # Apply augmentations sequentially
    augmented_problem = deepcopy(problem)
    applied_augmentations = []
    
    for aug_name in selected_augmentations:
        aug_func = available_augmentations[aug_name]
        
        # Handle special cases that need additional parameters
        if aug_name == 'color_permutation':
            # Generate a consistent color map for the entire problem
            colors = list(range(10))
            shuffled_colors = colors.copy()
            random.shuffle(shuffled_colors)
            color_map = dict(zip(colors, shuffled_colors))
            augmented_problem = apply_augmentation_to_problem(
                augmented_problem, aug_func, color_map=color_map
            )
        elif aug_name == 'upscale':
            # Use consistent target size for the entire problem
            # Get size of first input to determine reasonable target size
            first_input = None
            if 'train' in augmented_problem and augmented_problem['train']:
                first_input = augmented_problem['train'][0]['input']
            elif 'test' in augmented_problem and augmented_problem['test']:
                first_input = augmented_problem['test'][0]['input']
            
            if first_input:
                current_h, current_w = len(first_input), len(first_input[0])
                max_h = min(30, current_h + random.randint(1, max(1, 30 - current_h)))
                max_w = min(30, current_w + random.randint(1, max(1, 30 - current_w)))
                target_height = random.randint(current_h, max_h)
                target_width = random.randint(current_w, max_w)
                target_size = (target_height, target_width)
                
                augmented_problem = apply_augmentation_to_problem(
                    augmented_problem, aug_func, target_size=target_size
                )
                applied_augmentations.append(f"{aug_name}_{target_height}x{target_width}")
                continue
            else:
                # Fallback if no input found
                augmented_problem = apply_augmentation_to_problem(augmented_problem, aug_func)
                applied_augmentations.append(aug_name)
                continue
        else:
            augmented_problem = apply_augmentation_to_problem(augmented_problem, aug_func)
        
        applied_augmentations.append(aug_name)
    
    return augmented_problem, applied_augmentations


def create_augmented_dataset(problems: List[Dict[str, Any]], 
                           augmentations_per_problem: int = 1,
                           seed: int = None) -> List[Tuple[Dict[str, Any], List[str]]]:
    """
    Create an augmented dataset by applying random augmentations to multiple problems.
    
    Args:
        problems (List[Dict[str, Any]]): List of problems to augment
        augmentations_per_problem (int): Number of augmented versions to create per problem
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        List[Tuple[Dict[str, Any], List[str]]]: List of (augmented_problem, applied_augmentations) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    augmented_dataset = []
    
    for problem in problems:
        for _ in range(augmentations_per_problem):
            augmented_problem, applied_augs = apply_random_augmentations(problem)
            augmented_dataset.append((augmented_problem, applied_augs))
    
    return augmented_dataset


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test grid
    test_grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    print("Original grid:")
    for row in test_grid:
        print(row)
    
    print("\nRotate 90:")
    rotated = rotate_90(test_grid)
    for row in rotated:
        print(row)
    
    print("\nRotate 180:")
    rotated = rotate_180(test_grid)
    for row in rotated:
        print(row)
    
    print("\nFlip vertical:")
    flipped = flip_vertical(test_grid)
    for row in flipped:
        print(row)
    
    print("\nFlip horizontal:")
    flipped = flip_horizontal(test_grid)
    for row in flipped:
        print(row)
    
    print("\nColor permutation:")
    permuted = apply_color_permutation(test_grid)
    for row in permuted:
        print(row)
    
    print("\nUpscale to 5x5:")
    upscaled = upscale_grid(test_grid, (5, 5))
    for row in upscaled:
        print(row)
    
    # Test with a problem structure
    test_problem = {
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[5, 6], [7, 8]]},
            {'input': [[0, 1], [2, 3]], 'output': [[4, 5], [6, 7]]}
        ],
        'test': [
            {'input': [[8, 9], [1, 2]], 'output': [[3, 4], [5, 6]]}
        ]
    }
    
    print("\n" + "="*50)
    print("Testing random augmentations on a problem:")
    print("="*50)
    
    augmented_problem, applied_augs = apply_random_augmentations(test_problem, seed=42)
    print(f"Applied augmentations: {applied_augs}")
    print(f"Original train input: {test_problem['train'][0]['input']}")
    print(f"Augmented train input: {augmented_problem['train'][0]['input']}")
