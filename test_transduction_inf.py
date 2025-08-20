#!/usr/bin/env python3
"""
Simple test script to demonstrate ARC transduction inference.

This script tests the inference program on a few ARC problems to verify
that everything is working correctly.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transduction.inference import ARCTransductionInference
from transduction.metrics import check_array, check_value
from loader import load_evaluation_problem, list_evaluation_problems


def test_single_problem():
    """Test inference on a single problem."""
    print("=" * 60)
    print("TESTING SINGLE PROBLEM")
    print("=" * 60)
    
    try:
        # Initialize inference
        print("Initializing model...")
        inference = ARCTransductionInference(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        
        # Load a test problem
        problem_id = "00576224"  # We know this exists from the file structure
        print(f"Loading problem: {problem_id}")
        
        problem_data = load_evaluation_problem(problem_id)
        
        # Print problem info
        print(f"Train examples: {len(problem_data['train'])}")
        print(f"Test examples: {len(problem_data['test'])}")
        
        # Show the first training example
        print("\nFirst training example:")
        train_ex = problem_data['train'][0]
        print("Input:")
        for row in train_ex['input']:
            print(' '.join(map(str, row)))
        print("Output:")
        for row in train_ex['output']:
            print(' '.join(map(str, row)))
        
        # Perform inference
        print(f"\nPerforming inference...")
        result = inference.infer_single_problem(problem_data, verbose=True)
        
        print(f"\nResult: {'SUCCESS' if result['is_correct'] else 'FAILED'}")
        
    except Exception as e:
        print(f"Error during single problem test: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_problems():
    """Test inference on multiple problems."""
    print("=" * 60)
    print("TESTING MULTIPLE PROBLEMS")
    print("=" * 60)
    
    try:
        # Initialize inference
        print("Initializing model...")
        inference = ARCTransductionInference(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        
        # Get problem list
        problem_ids = list_evaluation_problems()
        print(f"Found {len(problem_ids)} evaluation problems")
        
        # Test on first 3 problems
        results = inference.evaluate_on_problems(
            problem_ids[:3],
            dataset_type="evaluation",
            train_sample_count=2  # Use fewer examples for faster testing
        )
        
        print(f"\nFinal Results:")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Correct: {results['correct_predictions']}/{results['total_problems']}")
        
    except Exception as e:
        print(f"Error during multiple problems test: {e}")
        import traceback
        traceback.print_exc()


def test_metrics():
    """Test the transduction metrics on ground truth data."""
    print("=" * 60)
    print("TESTING TRANSDUCTION METRICS")
    print("=" * 60)
    
    try:
        # Get list of evaluation problems
        problem_ids = list_evaluation_problems()
        print(f"Found {len(problem_ids)} evaluation problems")
        
        # Test on first 5 problems
        test_problems = problem_ids[:5]
        
        total_tests = 0
        check_array_passed = 0
        check_value_passed = 0
        
        for i, problem_id in enumerate(test_problems):
            print(f"\nTesting problem {i+1}/{len(test_problems)}: {problem_id}")
            
            try:
                # Load problem
                problem_data = load_evaluation_problem(problem_id)
                
                # Test each test case in the problem
                for test_idx, test_case in enumerate(problem_data['test']):
                    ground_truth = test_case['output']
                    
                    # Convert ground truth to semicolon format (expected model output format)
                    expected_output_str = ';'.join([''.join(map(str, row)) for row in ground_truth])
                    
                    print(f"  Test case {test_idx + 1}: shape {len(ground_truth)}x{len(ground_truth[0]) if ground_truth else 0}")
                    
                    # Test check_array
                    array_valid = check_array(expected_output_str)
                    
                    # Test check_value  
                    value_correct = check_value(expected_output_str, ground_truth)
                    
                    total_tests += 1
                    if array_valid:
                        check_array_passed += 1
                    if value_correct:
                        check_value_passed += 1
                    
                    status = "✅" if array_valid and value_correct else "❌"
                    print(f"    check_array: {'PASS' if array_valid else 'FAIL'}, check_value: {'PASS' if value_correct else 'FAIL'} {status}")
                        
            except Exception as e:
                print(f"  Error processing problem {problem_id}: {e}")
                continue
        
        print(f"\n" + "=" * 40)
        print("METRICS TEST SUMMARY")
        print("=" * 40)
        print(f"Total test cases: {total_tests}")
        print(f"check_array passed: {check_array_passed}/{total_tests} ({check_array_passed/total_tests*100:.1f}%)")
        print(f"check_value passed: {check_value_passed}/{total_tests} ({check_value_passed/total_tests*100:.1f}%)")
        
        if check_array_passed == total_tests and check_value_passed == total_tests:
            print("🎉 ALL METRICS TESTS PASSED!")
        else:
            print("❌ Some metric tests failed.")
            
    except Exception as e:
        print(f"Error during metrics testing: {e}")
        import traceback
        traceback.print_exc()


def test_metrics_with_model_output():
    """Test the metrics with actual model outputs."""
    print("=" * 60)
    print("TESTING METRICS WITH MODEL OUTPUT")
    print("=" * 60)
    
    print("This test demonstrates how the metrics work with actual model outputs.")
    print("Testing various output scenarios...")
    
    # Test cases: (model_output, expected_grid, description)
    test_cases = [
        ("12;34", [[1, 2], [3, 4]], "Perfect semicolon format"),
        ("The answer is: 12;34", [[1, 2], [3, 4]], "Output with extra text"),
        ("12;34 (final answer)", [[1, 2], [3, 4]], "Output with trailing text"),
        ("invalid response", [[1, 2], [3, 4]], "Invalid response"),
        ("12;35", [[1, 2], [3, 4]], "Wrong values"),
        ("12", [[1, 2]], "Single row output"),
        ("", [[1, 2]], "Empty response"),
        ("1a;23", [[1, 2], [3, 4]], "Invalid characters"),
    ]
    
    print(f"\nTesting {len(test_cases)} different scenarios:")
    
    for i, (model_output, expected_grid, description) in enumerate(test_cases):
        print(f"\n{i+1}. {description}")
        print(f"   Model output: '{model_output}'")
        print(f"   Expected grid: {expected_grid}")
        
        array_valid = check_array(model_output)
        value_correct = check_value(model_output, expected_grid)
        
        print(f"   check_array(): {'✅ PASS' if array_valid else '❌ FAIL'}")
        print(f"   check_value(): {'✅ PASS' if value_correct else '❌ FAIL'}")
        
        if array_valid and value_correct:
            print(f"   Result: 🎉 Model output is perfect!")
        elif array_valid and not value_correct:
            print(f"   Result: ⚠️ Model output is parseable but incorrect")
        else:
            print(f"   Result: ❌ Model output is invalid")


def main():
    """Main test function."""
    print("ARC Transduction Inference Test")
    print("This will test the inference program with Qwen-2.5-0.5B-Instruct")
    print()
    
    # Check if transformers is available
    try:
        import transformers
        import torch
        print(f"✓ transformers version: {transformers.__version__}")
        print(f"✓ torch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    print()
    
    # First test the metrics (fast and doesn't require model)
    print("Testing metrics first (no model required)...")
    test_metrics()
    test_metrics_with_model_output()
    
    print("\n" + "=" * 60)
    print("Would you like to test model inference? (This requires downloading a model)")
    response = input("Enter 'y' for yes, any other key to skip: ").lower().strip()
    
    if response == 'y':
        # Test single problem first
        test_single_problem()
        
        print("\n" + "=" * 60)
        print("Would you like to test multiple problems? (This will take longer)")
        response = input("Enter 'y' for yes, any other key to skip: ").lower().strip()
        
        if response == 'y':
            test_multiple_problems()
        else:
            print("Skipping multiple problems test.")
    else:
        print("Skipping model inference tests.")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
