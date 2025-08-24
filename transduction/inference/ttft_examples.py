"""
Examples of integrating TTFT with existing inference approaches.

This module demonstrates how to use TTFT with:
1. Simple inference (ARCTransductionInference)
2. AIRV inference
3. Multiple averaging approaches

TTFT can enhance any inference method by fine-tuning the model on each problem's
training data before performing inference.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transduction.inference.ttft import TTFTInference, TTFTWrapper, create_ttft_configs
from transduction.inference.inference import ARCTransductionInference
from transduction.inference.airv import AIRVInference
from loader import load_evaluation_problem, list_evaluation_problems


def example_simple_ttft(problem_id: str = "00576224", data_dir: str = "."):
    """
    Example of using TTFT with simple inference.
    
    Args:
        problem_id: ID of the problem to test
        data_dir: Directory containing ARC data
    """
    print("=" * 60)
    print("EXAMPLE 1: Simple TTFT Inference")
    print("=" * 60)
    
    # Load a test problem
    try:
        problem_data = load_evaluation_problem(problem_id, data_dir)
        print(f"Loaded problem {problem_id}")
        print(f"Training examples: {len(problem_data.get('train', []))}")
        print(f"Test examples: {len(problem_data.get('test', []))}")
    except Exception as e:
        print(f"Could not load problem {problem_id}: {e}")
        return
    
    # Create TTFT inference with light configuration
    ttft_config = {
        'model_name': "Qwen/Qwen2.5-0.5B-Instruct",
        'num_augmentations': 2,
        'ttft_epochs': 2,
        'ttft_learning_rate': 1e-4,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16
    }
    
    try:
        ttft = TTFTInference(**ttft_config)
        
        print("\nPerforming TTFT inference...")
        result = ttft.infer_single_problem(problem_data, verbose=True)
        
        print(f"\nResult: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
        print(f"Used {result.get('num_augmentations', 'N/A')} augmentations per example")
        print(f"Fine-tuned for {result.get('ttft_epochs', 'N/A')} epochs")
        
        ttft.cleanup()
        
    except Exception as e:
        print(f"Error during TTFT inference: {e}")


def example_ttft_with_airv(problem_id: str = "00576224", data_dir: str = "."):
    """
    Example of using TTFT with AIRV inference.
    
    Args:
        problem_id: ID of the problem to test
        data_dir: Directory containing ARC data
    """
    print("=" * 60)
    print("EXAMPLE 2: TTFT + AIRV Inference")
    print("=" * 60)
    
    # Load a test problem
    try:
        problem_data = load_evaluation_problem(problem_id, data_dir)
        print(f"Loaded problem {problem_id}")
    except Exception as e:
        print(f"Could not load problem {problem_id}: {e}")
        return
    
    # Create TTFT configuration
    ttft_config = {
        'model_name': "Qwen/Qwen2.5-0.5B-Instruct",
        'num_augmentations': 3,
        'ttft_epochs': 3,
        'ttft_learning_rate': 5e-5,
        'use_lora': True,
        'lora_r': 16,
        'lora_alpha': 32
    }
    
    try:
        # Create TTFT wrapper for AIRV
        ttft_wrapper = TTFTWrapper(AIRVInference, ttft_config)
        
        # Create AIRV inference enhanced with TTFT
        airv_with_ttft = ttft_wrapper.create_wrapped_inference(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            num_augmentations=4,  # AIRV's own augmentations
            include_original=True
        )
        
        print("\nPerforming TTFT + AIRV inference...")
        result = airv_with_ttft.infer_single_problem(problem_data, verbose=True)
        
        print(f"\nResult: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
        print(f"Used TTFT: {result.get('used_ttft', False)}")
        print(f"AIRV versions: {result.get('num_versions', 'N/A')}")
        print(f"Valid outputs: {result.get('valid_outputs_count', 'N/A')}")
        
        # Cleanup
        if hasattr(airv_with_ttft, 'cleanup'):
            airv_with_ttft.cleanup()
        
    except Exception as e:
        print(f"Error during TTFT + AIRV inference: {e}")


def example_multiple_ttft_configs(problem_id: str = "00576224", data_dir: str = "."):
    """
    Example of testing multiple TTFT configurations.
    
    Args:
        problem_id: ID of the problem to test
        data_dir: Directory containing ARC data
    """
    print("=" * 60)
    print("EXAMPLE 3: Multiple TTFT Configurations")
    print("=" * 60)
    
    # Load a test problem
    try:
        problem_data = load_evaluation_problem(problem_id, data_dir)
        print(f"Loaded problem {problem_id}")
    except Exception as e:
        print(f"Could not load problem {problem_id}: {e}")
        return
    
    # Get different TTFT configurations
    configs = create_ttft_configs()
    results = {}
    
    for config in configs[:2]:  # Test first 2 configs to save time
        print(f"\n--- Testing {config['name']}: {config['description']} ---")
        
        try:
            # Add model name to config
            full_config = config['params'].copy()
            full_config['model_name'] = "Qwen/Qwen2.5-0.5B-Instruct"
            
            ttft = TTFTInference(**full_config)
            result = ttft.infer_single_problem(problem_data, verbose=False)
            
            results[config['name']] = result
            
            print(f"Result: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
            print(f"Epochs: {result.get('ttft_epochs', 'N/A')}")
            print(f"Augmentations: {result.get('num_augmentations', 'N/A')}")
            print(f"LoRA: {result.get('use_lora', 'N/A')}")
            
            ttft.cleanup()
            
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Summary
    print(f"\n--- SUMMARY ---")
    correct_configs = [name for name, result in results.items() 
                      if result.get('is_correct', False)]
    
    print(f"Tested {len(results)} configurations")
    print(f"Correct predictions: {len(correct_configs)}")
    if correct_configs:
        print(f"Best configurations: {', '.join(correct_configs)}")


def example_ttft_averaging(problem_ids: List[str] = None, data_dir: str = ".", max_problems: int = 3):
    """
    Example of using TTFT with multiple problems for averaging results.
    
    Args:
        problem_ids: List of problem IDs to test (if None, uses first few from evaluation set)
        data_dir: Directory containing ARC data
        max_problems: Maximum number of problems to test
    """
    print("=" * 60)
    print("EXAMPLE 4: TTFT Multiple Problem Averaging")
    print("=" * 60)
    
    if problem_ids is None:
        try:
            all_problem_ids = list_evaluation_problems(data_dir)
            problem_ids = all_problem_ids[:max_problems]
            print(f"Using first {len(problem_ids)} problems from evaluation set")
        except Exception as e:
            print(f"Could not list problems: {e}")
            return
    
    # TTFT configuration
    ttft_config = {
        'model_name': "Qwen/Qwen2.5-0.5B-Instruct",
        'num_augmentations': 2,
        'ttft_epochs': 2,
        'ttft_learning_rate': 1e-4,
        'use_lora': True,
        'lora_r': 8,
        'lora_alpha': 16
    }
    
    results = []
    correct_count = 0
    
    try:
        ttft = TTFTInference(**ttft_config)
        
        for i, problem_id in enumerate(problem_ids):
            print(f"\n--- Problem {i+1}/{len(problem_ids)}: {problem_id} ---")
            
            try:
                problem_data = load_evaluation_problem(problem_id, data_dir)
                result = ttft.infer_single_problem(problem_data, verbose=False)
                
                results.append({
                    'problem_id': problem_id,
                    'is_correct': result['is_correct'],
                    'result': result
                })
                
                if result['is_correct']:
                    correct_count += 1
                    print("✓ CORRECT")
                else:
                    print("✗ INCORRECT")
                    
            except Exception as e:
                print(f"Error processing {problem_id}: {e}")
                results.append({
                    'problem_id': problem_id,
                    'is_correct': False,
                    'error': str(e)
                })
        
        ttft.cleanup()
        
    except Exception as e:
        print(f"Error initializing TTFT: {e}")
        return
    
    # Summary
    print(f"\n--- FINAL RESULTS ---")
    print(f"Total problems: {len(results)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {correct_count / len(results):.2%}")
    
    # Show individual results
    print(f"\nDetailed results:")
    for result in results:
        status = "✓" if result['is_correct'] else "✗"
        print(f"  {status} {result['problem_id']}")


def main():
    """Main function to run all examples."""
    print("TTFT Integration Examples")
    print("=" * 80)
    
    # Check if we have the required data directory
    data_dir = "."
    if not os.path.exists(os.path.join(data_dir, "evaluation_data")):
        print("Warning: evaluation_data directory not found.")
        print("These examples require ARC evaluation data to run.")
        print("Please make sure you're running from the correct directory.")
        return
    
    try:
        # Example 1: Simple TTFT
        example_simple_ttft()
        
        print("\n" + "=" * 80 + "\n")
        
        # Example 2: TTFT + AIRV (commented out as it requires more resources)
        # example_ttft_with_airv()
        # print("\n" + "=" * 80 + "\n")
        
        # Example 3: Multiple configurations
        example_multiple_ttft_configs()
        
        print("\n" + "=" * 80 + "\n")
        
        # Example 4: Multiple problems
        example_ttft_averaging()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\nExamples completed!")


if __name__ == "__main__":
    main()
