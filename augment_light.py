"""
Augmentation functions for ARC (Abstraction and Reasoning Corpus) problems.

This module provides various augmentation functions that can be applied to grid-based problems
to create variations for data augmentation purposes.
"""

import random
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
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
    }

def apply_random_augmentations(problem: Dict[str, Any], 
                             num_augmentations: int = None,
                             seed: int = None) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    
    if num_augmentations is None:
        num_augmentations = random.randint(2, 4)
    else:
        num_augmentations = max(2, min(4, num_augmentations))   
    available_augmentations = get_available_augmentations()
    augmentation_names = list(available_augmentations.keys())
    selected_augmentations = random.sample(augmentation_names, num_augmentations)
    augmented_problem = deepcopy(problem)
    
    for i, aug_name in enumerate(selected_augmentations):
        aug_id = f"{aug_name}_{i}"  # Unique identifier
        
        # Handle special cases that need tracking
        if aug_name == 'color_permutation':
            colors = list(range(10))
            shuffled_colors = colors.copy()
            random.shuffle(shuffled_colors)
            color_map = dict(zip(colors, shuffled_colors))
            if 'train' in augmented_problem:
                for example in augmented_problem['train']:
                    if 'input' in example:
                        example['input'] = [[color_map.get(cell, cell) for cell in row] 
                                          for row in example['input']]
                    if 'output' in example:
                        example['output'] = [[color_map.get(cell, cell) for cell in row] 
                                           for row in example['output']]
            if 'test' in augmented_problem:
                for example in augmented_problem['test']:
                    if 'input' in example:
                        example['input'] = [[color_map.get(cell, cell) for cell in row] 
                                          for row in example['input']]
                    if 'output' in example:
                        example['output'] = [[color_map.get(cell, cell) for cell in row] 
                                           for row in example['output']]

        else:
            aug_func = available_augmentations[aug_name]
            augmented_problem = apply_augmentation_to_problem(augmented_problem, aug_func)    
    return augmented_problem