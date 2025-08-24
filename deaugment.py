"""
Deaugmentation functions for ARC (Abstraction and Reasoning Corpus) problems.

This module provides inverse functions for each augmentation, allowing you to reverse
transformations that were applied to grid-based problems.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from copy import deepcopy


def derotate_90(grid: List[List[int]]) -> List[List[int]]:
    """
    Inverse of rotate_90: rotate 270 degrees clockwise (90 degrees counter-clockwise).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Derotated grid
    """
    if not grid or not grid[0]:
        return grid
    
    rows, cols = len(grid), len(grid[0])
    derotated = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            derotated[cols - 1 - j][i] = grid[i][j]
    
    return derotated


def derotate_180(grid: List[List[int]]) -> List[List[int]]:
    """
    Inverse of rotate_180: rotate 180 degrees (same operation).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Derotated grid
    """
    if not grid:
        return grid
    
    return [row[::-1] for row in grid[::-1]]


def derotate_270(grid: List[List[int]]) -> List[List[int]]:
    """
    Inverse of rotate_270: rotate 90 degrees clockwise.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Derotated grid
    """
    if not grid or not grid[0]:
        return grid
    
    rows, cols = len(grid), len(grid[0])
    derotated = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            derotated[j][rows - 1 - i] = grid[i][j]
    
    return derotated


def deflip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """
    Inverse of flip_vertical: flip vertically (same operation).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Deflipped grid
    """
    return grid[::-1]


def deflip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """
    Inverse of flip_horizontal: flip horizontally (same operation).
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        
    Returns:
        List[List[int]]: Deflipped grid
    """
    return [row[::-1] for row in grid]


def deapply_color_permutation(grid: List[List[int]], color_map: Dict[int, int]) -> List[List[int]]:
    """
    Inverse of apply_color_permutation: apply the inverse color mapping.
    
    Args:
        grid (List[List[int]]): Input grid as list of lists
        color_map (Dict[int, int]): Original color mapping used in augmentation
        
    Returns:
        List[List[int]]: Grid with inverse color permutation applied
    """
    # Create inverse mapping
    inverse_color_map = {v: k for k, v in color_map.items()}
    
    return [[inverse_color_map.get(cell, cell) for cell in row] for row in grid]


def deupscale_grid(grid: List[List[int]], 
                   original_size: Tuple[int, int],
                   original_position: Tuple[int, int] = None) -> List[List[int]]:
    """
    Inverse of upscale_grid: crop the grid back to original size.
    
    Args:
        grid (List[List[int]]): Input upscaled grid
        original_size (Tuple[int, int]): Original (height, width) before upscaling
        original_position (Tuple[int, int], optional): Position where original was placed.
                                                      If None, tries to find it automatically.
        
    Returns:
        List[List[int]]: Cropped grid
    """
    if not grid or not grid[0]:
        return grid
    
    orig_height, orig_width = original_size
    current_height, current_width = len(grid), len(grid[0])
    
    # If already the right size, return as is
    if current_height == orig_height and current_width == orig_width:
        return grid
    
    # Validate original size is reasonable
    if orig_height <= 0 or orig_width <= 0:
        print(f"Warning: Invalid original size {original_size}")
        return grid
    
    if original_position is not None:
        # Use provided position
        top_offset, left_offset = original_position
        # Clamp to valid bounds
        top_offset = max(0, min(top_offset, current_height - orig_height))
        left_offset = max(0, min(left_offset, current_width - orig_width))
    else:
        # Try to find the original content by looking for non-zero region
        # This is a heuristic and may not always work perfectly
        try:
            top_offset, left_offset = find_content_position(grid, original_size)
        except Exception as e:
            print(f"Warning: Failed to find content position: {e}")
            top_offset, left_offset = 0, 0
    
    # Extract the original region
    cropped = []
    for i in range(orig_height):
        row = []
        for j in range(orig_width):
            if (top_offset + i < current_height and 
                left_offset + j < current_width and
                top_offset + i >= 0 and left_offset + j >= 0):
                row.append(grid[top_offset + i][left_offset + j])
            else:
                row.append(0)  # Fallback if out of bounds
        cropped.append(row)
    
    return cropped


def find_content_position(grid: List[List[int]], original_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Try to find where the original content is located in an upscaled grid.
    This is a heuristic approach that looks for the densest non-zero region.
    
    Args:
        grid (List[List[int]]): Upscaled grid
        original_size (Tuple[int, int]): Size of original content
        
    Returns:
        Tuple[int, int]: (top_offset, left_offset) of likely original position
    """
    orig_height, orig_width = original_size
    current_height, current_width = len(grid), len(grid[0])
    
    # Validate inputs
    if orig_height > current_height or orig_width > current_width:
        print(f"Warning: Original size {original_size} larger than current size {current_height}x{current_width}")
        return (0, 0)
    
    best_score = -1
    best_position = (0, 0)
    
    # Try all possible positions
    max_top = max(0, current_height - orig_height + 1)
    max_left = max(0, current_width - orig_width + 1)
    
    for top in range(max_top):
        for left in range(max_left):
            # Count non-zero elements in this region
            score = 0
            try:
                for i in range(orig_height):
                    for j in range(orig_width):
                        if (top + i < current_height and left + j < current_width and 
                            grid[top + i][left + j] != 0):
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_position = (top, left)
            except IndexError:
                # Skip this position if out of bounds
                continue
    
    return best_position


def apply_deaugmentation_to_example(example: Dict[str, List[List[int]]], 
                                   deaugmentation_func: Callable,
                                   **kwargs) -> Dict[str, List[List[int]]]:
    """
    Apply a deaugmentation function to both input and output of an example.
    
    Args:
        example (Dict[str, List[List[int]]]): Example with 'input' and 'output' keys
        deaugmentation_func (Callable): Deaugmentation function to apply
        **kwargs: Additional arguments for the deaugmentation function
        
    Returns:
        Dict[str, List[List[int]]]: Deaugmented example
    """
    deaugmented_example = {}
    
    if 'input' in example:
        deaugmented_example['input'] = deaugmentation_func(example['input'], **kwargs)
    
    if 'output' in example:
        deaugmented_example['output'] = deaugmentation_func(example['output'], **kwargs)
    
    return deaugmented_example


def apply_deaugmentation_to_problem(problem: Dict[str, Any], 
                                   deaugmentation_func: Callable,
                                   **kwargs) -> Dict[str, Any]:
    """
    Apply a deaugmentation function to all examples in a problem.
    
    Args:
        problem (Dict[str, Any]): Problem dictionary with 'train', 'test', etc.
        deaugmentation_func (Callable): Deaugmentation function to apply
        **kwargs: Additional arguments for the deaugmentation function
        
    Returns:
        Dict[str, Any]: Problem with deaugmentation applied to all examples
    """
    deaugmented_problem = deepcopy(problem)
    
    # Apply to training examples
    if 'train' in deaugmented_problem:
        deaugmented_problem['train'] = [
            apply_deaugmentation_to_example(example, deaugmentation_func, **kwargs)
            for example in deaugmented_problem['train']
        ]
    
    # Apply to test examples
    if 'test' in deaugmented_problem:
        deaugmented_problem['test'] = [
            apply_deaugmentation_to_example(example, deaugmentation_func, **kwargs)
            for example in deaugmented_problem['test']
        ]
    
    # Apply to arc-gen examples if present
    if 'arc-gen' in deaugmented_problem:
        deaugmented_problem['arc-gen'] = [
            apply_deaugmentation_to_example(example, deaugmentation_func, **kwargs)
            for example in deaugmented_problem['arc-gen']
        ]
    
    return deaugmented_problem


def get_deaugmentation_functions() -> Dict[str, Callable]:
    """
    Get a dictionary of all available deaugmentation functions.
    
    Returns:
        Dict[str, Callable]: Dictionary mapping augmentation names to deaugmentation functions
    """
    return {
        'rotate_90': derotate_90,
        'rotate_180': derotate_180,
        'rotate_270': derotate_270,
        'flip_vertical': deflip_vertical,
        'flip_horizontal': deflip_horizontal,
        'color_permutation': deapply_color_permutation,
        'upscale': deupscale_grid
    }


def parse_augmentation_name(aug_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse an augmentation name to extract base name and parameters.
    
    Args:
        aug_name (str): Augmentation name (e.g., "upscale_8x10", "rotate_90")
        
    Returns:
        Tuple[str, Dict[str, Any]]: (base_name, parameters)
    """
    if '_' in aug_name:
        parts = aug_name.split('_', 1)  # Split only on first underscore
        base_name = parts[0]
        
        if base_name == 'upscale' and len(parts) > 1:
            # Parse size like "8x10"
            size_str = parts[1]
            if 'x' in size_str:
                height, width = map(int, size_str.split('x'))
                return base_name, {'target_size': (height, width)}
        
        # For rotate_90, rotate_180, etc., flip operations, and color_permutation, return the full name
        if base_name in ['rotate', 'flip', 'color']:
            return aug_name, {}  # Return full name like "rotate_90", "flip_vertical", "color_permutation"
        
        return base_name, {}
    else:
        return aug_name, {}


def apply_pixel_level_deaugmentation(predicted_grid: List[List[int]], 
                                   augmentation_metadata: Dict[str, Any]) -> List[List[int]]:
    """
    Apply pixel-level deaugmentation using comprehensive transformation metadata.
    This is more accurate than the sequential approach for complex transformations.
    
    Args:
        predicted_grid: The grid predicted on augmented data
        augmentation_metadata: Comprehensive metadata with pixel transformations
        
    Returns:
        Deaugmented grid in original space
    """
    if 'applied_augmentations' not in augmentation_metadata:
        return None
    
    applied_augmentations = augmentation_metadata['applied_augmentations']
    augmentation_params = augmentation_metadata.get('augmentation_params', {})
    
    # Start with the predicted grid and reverse each augmentation
    current_grid = [row[:] for row in predicted_grid]  # Deep copy
    
    # Apply deaugmentations in reverse order
    for aug_name in reversed(applied_augmentations):
        try:
            if aug_name == 'rotate_90':
                # Reverse rotate_90: apply rotate_270 (or 3 x rotate_90)
                current_grid = derotate_90(current_grid)
            elif aug_name == 'rotate_180':
                # Reverse rotate_180: apply rotate_180 again
                current_grid = derotate_180(current_grid)
            elif aug_name == 'rotate_270':
                # Reverse rotate_270: apply rotate_90
                current_grid = derotate_270(current_grid)
            elif aug_name == 'flip_vertical':
                # Reverse flip_vertical: apply flip_vertical again
                current_grid = deflip_vertical(current_grid)
            elif aug_name == 'flip_horizontal':
                # Reverse flip_horizontal: apply flip_horizontal again
                current_grid = deflip_horizontal(current_grid)
            elif aug_name == 'color_permutation':
                # Reverse color permutation
                if 'color_permutation' in augmentation_params and 'color_map' in augmentation_params['color_permutation']:
                    color_map = augmentation_params['color_permutation']['color_map']
                    # Convert string keys to integers if needed
                    if color_map and len(color_map) > 0:
                        # Safe way to check key type
                        first_key = next(iter(color_map.keys()))
                        if isinstance(first_key, str):
                            color_map = {int(k): int(v) for k, v in color_map.items()}
                    current_grid = deapply_color_permutation(current_grid, color_map)
            elif aug_name.startswith('upscale'):
                # Reverse upscaling
                if aug_name in augmentation_params:
                    params = augmentation_params[aug_name]
                    position_info = augmentation_metadata.get('position_transformations', {}).get(aug_name, {})
                    
                    # Get the size we need to crop back to
                    pre_upscale_size = params.get('pre_upscale_size', position_info.get('pre_upscale_size', params.get('original_size')))
                    
                    if pre_upscale_size is not None:
                        # Try to get stored offset, otherwise use heuristic detection
                        original_position = params.get('offset', position_info.get('offset', None))
                        current_grid = deupscale_grid(current_grid, pre_upscale_size, original_position)
                    else:
                        print(f"Warning: No size information for deaugmenting {aug_name}")
        except Exception as e:
            print(f"Warning: Error deaugmenting {aug_name}: {e}")
            # Continue with other deaugmentations
            continue
    
    return current_grid


def apply_full_deaugmentation(problem: Dict[str, Any], 
                             augmentation_list: List[str],
                             augmentation_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Apply full deaugmentation by reversing a list of augmentations in reverse order.
    
    Args:
        problem (Dict[str, Any]): Augmented problem to deaugment
        augmentation_list (List[str]): List of augmentations that were applied (in order)
        augmentation_metadata (Dict[str, Any], optional): Metadata about augmentations
                                                         (e.g., color maps, original sizes)
        
    Returns:
        Dict[str, Any]: Deaugmented problem
    """
    deaugmentation_funcs = get_deaugmentation_functions()
    deaugmented_problem = deepcopy(problem)
    
    if augmentation_metadata is None:
        augmentation_metadata = {}
    
    # Apply deaugmentations in reverse order
    for aug_name in reversed(augmentation_list):
        try:
            base_name, params = parse_augmentation_name(aug_name)
            
            if base_name not in deaugmentation_funcs:
                print(f"Warning: No deaugmentation function for '{base_name}'")
                continue
            
            deaug_func = deaugmentation_funcs[base_name]
            
            # Handle special cases that need additional parameters
            if base_name == 'color_permutation':
                # Need the original color map - try multiple possible locations
                color_map = None
                
                # First try the augmentation_params structure (most common)
                if 'augmentation_params' in augmentation_metadata:
                    aug_params = augmentation_metadata['augmentation_params']
                    if 'color_permutation' in aug_params:
                        color_map = aug_params['color_permutation'].get('color_map')
                    elif aug_name in aug_params:
                        color_map = aug_params[aug_name].get('color_map')
                
                # Fallback to older metadata structure
                if not color_map:
                    if 'color_permutation' in augmentation_metadata:
                        color_map = augmentation_metadata['color_permutation'].get('color_map')
                    elif 'color_map' in augmentation_metadata:
                        color_map = augmentation_metadata['color_map']
                
                if color_map:
                    deaugmented_problem = apply_deaugmentation_to_problem(
                        deaugmented_problem, deaug_func, color_map=color_map
                    )
                else:
                    print(f"Warning: No color_map provided for deaugmenting {aug_name}")
                    print(f"Available metadata keys: {list(augmentation_metadata.keys())}")
                    if 'augmentation_params' in augmentation_metadata:
                        print(f"Available augmentation_params keys: {list(augmentation_metadata['augmentation_params'].keys())}")
            elif base_name == 'upscale':
                # Need original size and optionally original position
                # Try multiple possible locations for the parameters
                original_size = None
                original_position = None
                
                # First try the augmentation_params structure (most common)
                if 'augmentation_params' in augmentation_metadata:
                    aug_params = augmentation_metadata['augmentation_params']
                    if aug_name in aug_params:
                        upscale_params = aug_params[aug_name]
                        original_size = upscale_params.get('original_size') or upscale_params.get('pre_upscale_size')
                        original_position = upscale_params.get('offset') or upscale_params.get('original_position')
                
                # Fallback to older metadata structure
                if not original_size:
                    if aug_name in augmentation_metadata:
                        upscale_params = augmentation_metadata[aug_name]
                        original_size = upscale_params.get('original_size') or upscale_params.get('pre_upscale_size')
                        original_position = upscale_params.get('offset') or upscale_params.get('original_position')
                    elif 'original_size' in augmentation_metadata:
                        original_size = augmentation_metadata['original_size']
                        original_position = augmentation_metadata.get('original_position')
                
                if original_size:
                    deaugmented_problem = apply_deaugmentation_to_problem(
                        deaugmented_problem, deaug_func, 
                        original_size=original_size,
                        original_position=original_position
                    )
                else:
                    print(f"Warning: No original_size provided for deaugmenting {aug_name}")
                    print(f"Available metadata keys: {list(augmentation_metadata.keys())}")
                    if 'augmentation_params' in augmentation_metadata:
                        print(f"Available augmentation_params keys: {list(augmentation_metadata['augmentation_params'].keys())}")
            else:
                # Simple deaugmentation without extra parameters
                deaugmented_problem = apply_deaugmentation_to_problem(
                    deaugmented_problem, deaug_func
                )
        except Exception as e:
            print(f"Warning: Error deaugmenting {aug_name}: {e}")
            continue
    
    return deaugmented_problem


def create_augmentation_metadata(original_problem: Dict[str, Any],
                               augmentation_list: List[str]) -> Dict[str, Any]:
    """
    Create metadata needed for deaugmentation based on original problem and augmentations.
    
    Args:
        original_problem (Dict[str, Any]): Original problem before augmentation
        augmentation_list (List[str]): List of augmentations that will be applied
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    metadata = {}
    
    # Get original size from first input
    if 'train' in original_problem and original_problem['train']:
        first_input = original_problem['train'][0]['input']
        metadata['original_size'] = (len(first_input), len(first_input[0]))
    
    # If color_permutation is in the list, we'd need to store the color map
    # This would need to be done during actual augmentation
    
    return metadata


# Example usage and testing
if __name__ == "__main__":
    # Test deaugmentation functions
    test_grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    def print_grid(grid, title="Grid"):
        """Helper function to print a grid nicely."""
        print(f"{title}:")
        for row in grid:
            print(' '.join(f"{cell:2d}" for cell in row))
        print()
    
    print_grid(test_grid, "Original grid")
    
    # Test rotation and derotation
    from augment import rotate_90
    rotated = rotate_90(test_grid)
    print_grid(rotated, "After rotate_90")
    
    derotated = derotate_90(rotated)
    print_grid(derotated, "After derotate_90")
    
    # Test upscale and deupscale
    from augment import upscale_grid
    upscaled = upscale_grid(test_grid, (5, 5))
    print_grid(upscaled, "After upscale to 5x5")
    
    deupscaled = deupscale_grid(upscaled, (3, 3))
    print_grid(deupscaled, "After deupscale back to 3x3")
    
    # Test full deaugmentation
    print("\n" + "="*50)
    print("Testing full deaugmentation:")
    print("="*50)
    
    # Simulate augmentation sequence
    augmentation_sequence = ['rotate_90', 'flip_horizontal', 'upscale_5x5']
    
    # Apply augmentations manually for testing
    step1 = rotate_90(test_grid)
    from augment import flip_horizontal
    step2 = flip_horizontal(step1)
    step3 = upscale_grid(step2, (5, 5))
    
    print_grid(step3, "After all augmentations")
    
    # Create test problem
    test_problem = {
        'train': [{'input': step3, 'output': step3}]
    }
    
    # Create metadata
    metadata = {
        'original_size': (3, 3),
        'original_position': None  # Will be auto-detected
    }
    
    # Apply full deaugmentation
    deaugmented = apply_full_deaugmentation(test_problem, augmentation_sequence, metadata)
    
    print_grid(deaugmented['train'][0]['input'], "After full deaugmentation")
    print(f"Matches original: {deaugmented['train'][0]['input'] == test_grid}")
