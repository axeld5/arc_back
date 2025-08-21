"""
Loader functions for ARC (Abstraction and Reasoning Corpus) problems.

This module provides functions to load problems from the evaluation and training datasets.
Problems are stored as JSON files with unique IDs.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_evaluation_problem(problem_id: str, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the evaluation dataset given its ID.
    
    Args:
        problem_id (str): The unique identifier for the problem (e.g., "00576224")
        data_dir (str): The root directory containing the evaluation folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        - "arc-gen": (if present) Additional generated examples
        
    Raises:
        FileNotFoundError: If the evaluation directory or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
    """
    evaluation_dir = Path(data_dir) / "evaluation_data"
    problem_file = evaluation_dir / f"{problem_id}.json"
    
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        return problem_data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")


def load_training_problem(problem_id: str, data_dir: str = ".") -> Optional[Dict[str, Any]]:
    """
    Load a problem from the training dataset given its ID.
    
    Args:
        problem_id (str): The unique identifier for the problem (e.g., "007bbfb7")
        data_dir (str): The root directory containing the training folder. Defaults to current directory.
        
    Returns:
        Optional[Dict[str, Any]]: The problem data as a dictionary, or None if not found.
        
    The returned dictionary contains:
        - "train": List of training examples, each with "input" and "output" grids
        - "test": List of test examples, each with "input" and "output" grids
        - "arc-gen": (if present) Additional generated examples
        
    Raises:
        FileNotFoundError: If the training directory or problem file doesn't exist
        json.JSONDecodeError: If the problem file contains invalid JSON
    """
    training_dir = Path(data_dir) / "training_data"
    problem_file = training_dir / f"{problem_id}.json"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
    
    try:
        with open(problem_file, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        return problem_data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in problem file {problem_file}: {e}")


def list_evaluation_problems(data_dir: str = ".") -> List[str]:
    """
    List all available problem IDs in the evaluation dataset.
    
    Args:
        data_dir (str): The root directory containing the evaluation folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of problem IDs (without the .json extension)
        
    Raises:
        FileNotFoundError: If the evaluation directory doesn't exist
    """
    evaluation_dir = Path(data_dir) / "evaluation"
    
    if not evaluation_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")
    
    problem_files = evaluation_dir.glob("*.json")
    return [f.stem for f in problem_files]


def list_training_problems(data_dir: str = ".") -> List[str]:
    """
    List all available problem IDs in the training dataset.
    
    Args:
        data_dir (str): The root directory containing the training folder. Defaults to current directory.
        
    Returns:
        List[str]: A list of problem IDs (without the .json extension)
        
    Raises:
        FileNotFoundError: If the training directory doesn't exist
    """
    training_dir = Path(data_dir) / "training"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    problem_files = training_dir.glob("*.json")
    return [f.stem for f in problem_files]


def get_problem_stats(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about a problem.
    
    Args:
        problem_data (Dict[str, Any]): The problem data dictionary
        
    Returns:
        Dict[str, Any]: Statistics including number of training/test examples and grid sizes
    """
    stats = {
        "train_examples": len(problem_data.get("train", [])),
        "test_examples": len(problem_data.get("test", [])),
        "arc_gen_examples": len(problem_data.get("arc-gen", [])),
    }
    
    # Get grid size information from first training example if available
    if problem_data.get("train"):
        first_example = problem_data["train"][0]
        if "input" in first_example:
            input_grid = first_example["input"]
            stats["input_height"] = len(input_grid)
            stats["input_width"] = len(input_grid[0]) if input_grid else 0
        
        if "output" in first_example:
            output_grid = first_example["output"]
            stats["output_height"] = len(output_grid)
            stats["output_width"] = len(output_grid[0]) if output_grid else 0
    
    return stats


# Example usage
if __name__ == "__main__":
    # Example: Load an evaluation problem
    try:
        problem = load_evaluation_problem("00576224")
        if problem:
            print("Loaded evaluation problem successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
            # Show stats
            stats = get_problem_stats(problem)
            print(f"Problem stats: {stats}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: Load a training problem
    try:
        problem = load_training_problem("007bbfb7")
        if problem:
            print("\nLoaded training problem successfully!")
            print(f"Training examples: {len(problem['train'])}")
            print(f"Test examples: {len(problem['test'])}")
            
            # Show stats
            stats = get_problem_stats(problem)
            print(f"Problem stats: {stats}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: List available problems
    try:
        eval_problems = list_evaluation_problems()
        print(f"\nFound {len(eval_problems)} evaluation problems")
        
        train_problems = list_training_problems()
        print(f"Found {len(train_problems)} training problems")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
