"""
Data generation for transduction task in ARC.

This module creates training datasets by sampling from training problems and formatting them
with the transduction prompt for training language models.

Key features:
- Leave-one-out sampling from training examples
- Diverse test placeholder generation strategies:
  * 1/4 probability: All zeros (original behavior)
  * 1/4 probability: Use a matrix from problem data (input/output from train/test)
  * 1/4 probability: Use a matrix but modify 1-30 random pixels
  * 1/4 probability: Use the ground-truth test output as placeholder
- Optional data augmentation with rotations, flips, color permutations, and upscaling
"""

import json
import random
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from copy import deepcopy

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader import list_training_problems, load_training_problem, list_evaluation_problems, load_evaluation_problem
from augment import apply_random_augmentations, get_available_augmentations
from transduction.prompts import PROMPT_V1


def grid_to_row_strings(grid: List[List[int]]) -> List[str]:
    """
    Convert a grid (list of lists) to row-string format.
    
    Args:
        grid: Grid as list of lists of integers
        
    Returns:
        List of strings, where each string represents a row
    """
    return [''.join(map(str, row)) for row in grid]


def format_train_examples(train_examples: List[Dict[str, List[List[int]]]]) -> str:
    """
    Format training examples for the transduction prompt.
    
    Args:
        train_examples: List of training examples with 'input' and 'output' keys
        
    Returns:
        Formatted string for TRAIN section of prompt
    """
    formatted_pairs = []
    for i, example in enumerate(train_examples, 1):
        input_rows = grid_to_row_strings(example['input'])
        output_rows = grid_to_row_strings(example['output'])
        
        input_str = ';'.join(input_rows)
        output_str = ';'.join(output_rows)
        
        formatted_pairs.append(f"TRAIN INPUT {i}: {input_str}\nTRAIN OUTPUT {i}: {output_str}")
    
    return '\n\n'.join(formatted_pairs)


def create_test_placeholder(problem_data: Dict[str, Any], 
                          sampled_train: List[Dict[str, List[List[int]]]], 
                          test_example: Dict[str, List[List[int]]]) -> str:
    """
    Create test placeholder rows with diverse strategies.
    
    Args:
        problem_data: Full problem data
        sampled_train: Sampled training examples
        test_example: Current test example
        
    Returns:
        Placeholder string in semicolon-separated format
    """
    target_height = len(test_example['input'])
    target_width = len(test_example['input'][0]) if target_height > 0 else 0
    
    # Collect all available matrices from the problem
    all_matrices = []
    
    # Add from sampled training examples
    for example in sampled_train:
        all_matrices.append(example['input'])
        all_matrices.append(example['output'])
    
    # Add from test examples (inputs only, not outputs since we don't want to leak)
    test_examples = problem_data.get('test', [])
    for example in test_examples:
        all_matrices.append(example['input'])
    
    # Add from other training examples not in sampled_train
    all_train_examples = []
    if 'train' in problem_data:
        all_train_examples.extend(problem_data['train'])
    if 'arc-gen' in problem_data:
        all_train_examples.extend(problem_data['arc-gen'])
    
    for example in all_train_examples:
        if example not in sampled_train:
            all_matrices.append(example['input'])
            all_matrices.append(example['output'])
    
    # Filter matrices that match target dimensions
    compatible_matrices = []
    for matrix in all_matrices:
        if len(matrix) == target_height and (not matrix or len(matrix[0]) == target_width):
            compatible_matrices.append(matrix)
    
    # Choose strategy with specified probabilities (1/4 each)
    strategy = random.choices(['zeros', 'matrix', 'modified_matrix', 'ground_truth'], weights=[1/4, 1/4, 1/4, 1/4])[0]
    
    if strategy == 'zeros' or (strategy in ['matrix', 'modified_matrix'] and not compatible_matrices):
        # Strategy 1: All zeros
        placeholder_matrix = [['0'] * target_width for _ in range(target_height)]
    
    elif strategy == 'matrix':
        # Strategy 2: Use a matrix from problem data
        chosen_matrix = random.choice(compatible_matrices)
        placeholder_matrix = [[str(cell) for cell in row] for row in chosen_matrix]
    
    elif strategy == 'modified_matrix':
        # Strategy 3: Use matrix but modify 1-30 random pixels
        chosen_matrix = random.choice(compatible_matrices)
        placeholder_matrix = [[str(cell) for cell in row] for row in chosen_matrix]
        
        # Modify random pixels
        num_modifications = random.randint(1, 30)
        total_pixels = target_height * target_width
        
        # Don't modify more pixels than exist
        num_modifications = min(num_modifications, total_pixels)
        
        # Generate random positions to modify
        positions = [(i, j) for i in range(target_height) for j in range(target_width)]
        positions_to_modify = random.sample(positions, num_modifications)
        
        for i, j in positions_to_modify:
            # Random color 0-9
            placeholder_matrix[i][j] = str(random.randint(0, 9))

    else:  # strategy == 'ground_truth'
        # Strategy 4: Use the ground-truth output as placeholder
        gt = test_example.get('output')
        if gt is not None and len(gt) == target_height and (not gt or len(gt[0]) == target_width):
            placeholder_matrix = [[str(cell) for cell in row] for row in gt]
        else:
            # Fallback to zeros if dimensions mismatch
            placeholder_matrix = [['0'] * target_width for _ in range(target_height)]
    
    # Convert to semicolon-separated format
    placeholder_rows = [''.join(row) for row in placeholder_matrix]
    return ';'.join(placeholder_rows)


def create_transduction_sample(problem_data: Dict[str, Any], 
                              train_sample_count: int, 
                              test_example_idx: int = 0) -> Dict[str, str]:
    """
    Create a single transduction training sample from a problem.
    
    Args:
        problem_data: Problem data with 'train', 'test', and optionally 'arc-gen' keys
        train_sample_count: Number of training examples to sample (2-4)
        test_example_idx: Index of test example to use as the target
        
    Returns:
        Dictionary with 'input' (prompt) and 'output' (target) keys
    """
    # Collect all available training examples from train + arc-gen
    all_train_examples = []
    
    if 'train' in problem_data:
        all_train_examples.extend(problem_data['train'])
    
    if 'arc-gen' in problem_data:
        all_train_examples.extend(problem_data['arc-gen'])
    
    # Sample training examples
    if len(all_train_examples) >= train_sample_count:
        sampled_train = random.sample(all_train_examples, train_sample_count)
    else:
        # If not enough examples, use all available and pad with repetition if needed
        sampled_train = all_train_examples.copy()
        while len(sampled_train) < train_sample_count:
            sampled_train.extend(random.sample(all_train_examples, 
                                             min(len(all_train_examples), 
                                                 train_sample_count - len(sampled_train))))
        sampled_train = sampled_train[:train_sample_count]
    
    # Get test example
    test_examples = problem_data.get('test', [])
    if not test_examples:
        raise ValueError("No test examples available in problem")
    
    test_example = test_examples[test_example_idx % len(test_examples)]
    
    # Format the prompt
    train_pairs_formatted = format_train_examples(sampled_train)
    test_input_formatted = grid_to_row_strings(test_example['input'])
    test_input_str = ';'.join(test_input_formatted)
    
    # Create diverse test placeholder using new strategy
    test_placeholder_str = create_test_placeholder(problem_data, sampled_train, test_example)
    
    prompt = PROMPT_V1.format(
        train_pairs=train_pairs_formatted,
        test_input=test_input_str,
        test_output_placeholder=test_placeholder_str
    )
    
    # Target output
    target_output = grid_to_row_strings(test_example['output'])
    target_output_str = ';'.join(target_output)
    
    return {
        'input': prompt,
        'output': target_output_str  # Semicolon-separated format
    }


def sample_problem_multiple_times(problem_data: Dict[str, Any], 
                                k: int = 1,
                                num_augmentations: int = None) -> List[Dict[str, str]]:
    """
    Sample a problem k times with different configurations and optionally apply augmentations.
    
    Args:
        problem_data: Problem data
        k: Number of samples to generate from this problem
        num_augmentations: Number of augmentations to apply (0-4), if None then random
        
    Returns:
        List of training samples
    """
    samples = []
    
    for _ in range(k):
        # Apply augmentations if specified
        if num_augmentations is not None and num_augmentations > 0:
            augmented_problem, _, _ = apply_random_augmentations(
                problem_data, 
                num_augmentations=num_augmentations
            )
        else:
            augmented_problem = problem_data
        
        # Random number of training examples (2-4)
        train_sample_count = random.randint(2, 4)
        
        # Random test example (if multiple available)
        test_examples = augmented_problem.get('test', [])
        test_idx = random.randint(0, len(test_examples) - 1) if test_examples else 0
        
        try:
            sample = create_transduction_sample(
                augmented_problem, 
                train_sample_count, 
                test_idx
            )
            samples.append(sample)
        except (ValueError, IndexError) as e:
            # Skip problematic samples
            print(f"Warning: Skipping sample due to error: {e}")
            continue
    
    return samples


def generate_transduction_dataset(data_dir: str = ".", 
                                k_per_problem: int = 3,
                                max_problems: int = None,
                                apply_augmentations: bool = True) -> List[Dict[str, str]]:
    """
    Generate the full transduction dataset.
    
    Args:
        data_dir: Directory containing training data
        k_per_problem: Number of samples to generate per problem
        max_problems: Maximum number of problems to process (None for all)
        apply_augmentations: Whether to apply random augmentations
        
    Returns:
        List of training samples
    """
    print("Loading training problems...")
    training_problem_ids = list_training_problems(data_dir)
    
    if max_problems:
        training_problem_ids = training_problem_ids[:max_problems]
    
    print(f"Processing {len(training_problem_ids)} training problems...")
    
    all_samples = []
    
    for i, problem_id in enumerate(training_problem_ids):
        if i % 50 == 0:
            print(f"Processing problem {i+1}/{len(training_problem_ids)}: {problem_id}")
        
        try:
            problem_data = load_training_problem(problem_id, data_dir)
            
            # Random number of augmentations (0-4) if augmentations are enabled
            num_augmentations = None
            if apply_augmentations:
                num_augmentations = random.randint(0, 4)
            
            problem_samples = sample_problem_multiple_times(
                problem_data, 
                k=k_per_problem,
                num_augmentations=num_augmentations
            )
            
            all_samples.extend(problem_samples)
            
        except Exception as e:
            print(f"Error processing problem {problem_id}: {e}")
            continue
    
    print(f"Generated {len(all_samples)} training samples")
    return all_samples


def save_dataset(samples: List[Dict[str, str]], output_file: str):
    """
    Save the dataset in HuggingFace-compatible format.
    
    Args:
        samples: List of training samples
        output_file: Output file path
    """
    print(f"Saving {len(samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved successfully!")


def main():
    """Main execution function for generating the transduction dataset."""
    parser = argparse.ArgumentParser(description='Generate transduction training dataset for ARC')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Directory containing training data (default: current directory)')
    parser.add_argument('--output', type=str, default='transduction/train_dataset.json',
                       help='Output file path (default: transduction/train_dataset.json)')
    parser.add_argument('--k_per_problem', type=int, default=3,
                       help='Number of samples to generate per problem (default: 3)')
    parser.add_argument('--max_problems', type=int, default=None,
                       help='Maximum number of problems to process (default: all)')
    parser.add_argument('--no_augmentations', action='store_true',
                       help='Disable random augmentations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate dataset
    samples = generate_transduction_dataset(
        data_dir=args.data_dir,
        k_per_problem=args.k_per_problem,
        max_problems=args.max_problems,
        apply_augmentations=not args.no_augmentations
    )
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    save_dataset(samples, args.output)
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(samples)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
