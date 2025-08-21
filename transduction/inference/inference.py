"""
Inference module for ARC transduction using language models.

This module performs direct inference on ARC problems using the transduction prompt
without relying on a pre-generated dataset. It loads problems directly and formats
them with the prompt for inference.
"""

import json
import os
import sys
import argparse
import random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from loader import load_evaluation_problem, load_training_problem, list_evaluation_problems, list_training_problems
from transduction.prompts import PROMPT_V1
from transduction.data_gen import grid_to_row_strings, format_train_examples

# HuggingFace imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers and/or torch not available. Please install with: pip install transformers torch")
    HF_AVAILABLE = False


class ARCTransductionInference:
    """
    Class for performing ARC transduction inference using language models.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto"):
        """
        Initialize the inference class with a language model.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required for inference")
        
        self.model_name = model_name
        self.device = self._get_device(device)
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True
        )
        
        # Move to device if using CPU
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("Model loaded successfully!")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def format_problem_for_inference(self, problem_data: Dict[str, Any], 
                                   train_sample_count: int = 3,
                                   test_example_idx: int = 0) -> str:
        """
        Format an ARC problem for inference using the transduction prompt.
        
        Args:
            problem_data: Problem data with 'train' and 'test' keys
            train_sample_count: Number of training examples to use (2-4)
            test_example_idx: Index of test example to use
            
        Returns:
            Formatted prompt string
        """
        # Get training examples
        train_examples = problem_data.get('train', [])
        if len(train_examples) < train_sample_count:
            # Use all available if not enough
            sampled_train = train_examples
        else:
            # Sample the requested number
            sampled_train = train_examples[:train_sample_count]
        
        # Get test example
        test_examples = problem_data.get('test', [])
        if not test_examples:
            raise ValueError("No test examples available in problem")
        
        test_example = test_examples[test_example_idx % len(test_examples)]
        
        # Format the prompt
        train_pairs_formatted = format_train_examples(sampled_train)
        test_input_formatted = grid_to_row_strings(test_example['input'])
        test_input_str = ';'.join(test_input_formatted)
        
        # Create a placeholder with same dimensions as test input
        test_placeholder_rows = ['0' * len(row) for row in test_input_formatted]
        test_placeholder_str = ';'.join(test_placeholder_rows)
        
        prompt = PROMPT_V1.format(
            train_pairs=train_pairs_formatted,
            test_input=test_input_str,
            test_output_placeholder=test_placeholder_str
        )
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model given a prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated response string
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response (only the new tokens)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def parse_grid_response(self, response: str) -> Optional[List[List[int]]]:
        """
        Parse the model's response to extract a grid.
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed grid as list of lists, or None if parsing fails
        """
        # Clean the response
        response = response.strip()
        
        # Look for semicolon-separated format first
        if ';' in response:
            # Extract the part that looks like a grid (digits and semicolons)
            import re
            grid_match = re.search(r'[0-9;]+', response)
            if grid_match:
                grid_str = grid_match.group()
                try:
                    rows = grid_str.split(';')
                    grid = []
                    for row in rows:
                        if row.strip():  # Skip empty rows
                            grid_row = [int(char) for char in row if char.isdigit()]
                            if grid_row:  # Only add non-empty rows
                                grid.append(grid_row)
                    
                    if grid:  # Only return if we got a valid grid
                        return grid
                except (ValueError, IndexError):
                    pass
        
        # Fallback: try to extract any sequence of digits
        import re
        digits = re.findall(r'\d', response)
        if digits:
            # Try to infer grid structure (this is a heuristic)
            # For now, return None if we can't parse semicolon format
            pass
        
        return None
    
    def evaluate_prediction(self, prediction: Optional[List[List[int]]], 
                          ground_truth: List[List[int]]) -> bool:
        """
        Evaluate if a prediction matches the ground truth.
        
        Args:
            prediction: Predicted grid
            ground_truth: Ground truth grid
            
        Returns:
            True if prediction matches ground truth exactly
        """
        if prediction is None:
            return False
        
        return prediction == ground_truth
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Perform inference on a single ARC problem.
        
        Args:
            problem_data: Problem data dictionary
            train_sample_count: Number of training examples to use
            test_example_idx: Index of test example to use
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with inference results
        """
        # Format prompt
        prompt = self.format_problem_for_inference(
            problem_data, train_sample_count, test_example_idx
        )
        
        if verbose:
            print(f"Prompt length: {len(prompt)} characters")
            print("=" * 50)
            print("PROMPT:")
            print(prompt)
            print("=" * 50)
        
        # Generate response
        response = self.generate_response(prompt)
        
        if verbose:
            print("MODEL RESPONSE:")
            print(response)
            print("=" * 50)
        
        # Parse response
        predicted_grid = self.parse_grid_response(response)
        
        # Get ground truth
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        
        # Evaluate
        is_correct = self.evaluate_prediction(predicted_grid, ground_truth)
        
        result = {
            'prompt': prompt,
            'response': response,
            'predicted_grid': predicted_grid,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'train_sample_count': train_sample_count,
            'test_example_idx': test_example_idx
        }
        
        if verbose:
            print(f"GROUND TRUTH:")
            for row in ground_truth:
                print(';'.join(map(str, row)))
            print()
            
            print(f"PREDICTED:")
            if predicted_grid:
                for row in predicted_grid:
                    print(';'.join(map(str, row)))
            else:
                print("Failed to parse prediction")
            print()
            
            print(f"CORRECT: {is_correct}")
            print("=" * 50)
        
        return result
    
    def evaluate_on_problems(self, problem_ids: List[str], 
                           data_dir: str = ".",
                           dataset_type: str = "evaluation",
                           max_problems: Optional[int] = None,
                           train_sample_count: int = 3) -> Dict[str, Any]:
        """
        Evaluate the model on a list of problems.
        
        Args:
            problem_ids: List of problem IDs to evaluate
            data_dir: Directory containing the data
            dataset_type: "evaluation" or "training"
            max_problems: Maximum number of problems to evaluate
            train_sample_count: Number of training examples to use per problem
            
        Returns:
            Dictionary with evaluation results
        """
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        
        print(f"Evaluating on {len(problem_ids)} problems from {dataset_type} set...")
        
        results = []
        correct_count = 0
        
        for i, problem_id in enumerate(problem_ids):
            print(f"\nProblem {i+1}/{len(problem_ids)}: {problem_id}")
            
            try:
                # Load problem
                if dataset_type == "evaluation":
                    problem_data = load_evaluation_problem(problem_id, data_dir)
                else:
                    problem_data = load_training_problem(problem_id, data_dir)
                
                # Perform inference
                result = self.infer_single_problem(
                    problem_data, 
                    train_sample_count=train_sample_count,
                    verbose=False
                )
                
                result['problem_id'] = problem_id
                results.append(result)
                
                if result['is_correct']:
                    correct_count += 1
                    print(f"✓ Correct")
                else:
                    print(f"✗ Incorrect")
                
            except Exception as e:
                print(f"Error processing problem {problem_id}: {e}")
                continue
        
        accuracy = correct_count / len(results) if results else 0
        
        evaluation_results = {
            'problem_results': results,
            'total_problems': len(results),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'model_name': self.model_name,
            'train_sample_count': train_sample_count
        }
        
        print(f"\nEvaluation Results:")
        print(f"Total problems: {len(results)}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.3f}")
        
        return evaluation_results


def main():
    """Main function for running ARC transduction inference."""
    parser = argparse.ArgumentParser(description='ARC Transduction Inference')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help='HuggingFace model name (default: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--device', type=str, default="auto",
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--data_dir', type=str, default=".",
                       help='Directory containing ARC data')
    parser.add_argument('--dataset', type=str, choices=['evaluation', 'training'], 
                       default='evaluation',
                       help='Dataset to evaluate on')
    parser.add_argument('--problem_id', type=str, default=None,
                       help='Specific problem ID to test (if not provided, tests multiple)')
    parser.add_argument('--max_problems', type=int, default=10,
                       help='Maximum number of problems to evaluate')
    parser.add_argument('--train_samples', type=int, default=3,
                       help='Number of training examples to use per problem')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (optional)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    if HF_AVAILABLE:
        torch.manual_seed(args.seed)
    
    # Initialize inference class
    try:
        inference = ARCTransductionInference(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    if args.problem_id:
        # Test single problem
        print(f"Testing single problem: {args.problem_id}")
        
        try:
            if args.dataset == "evaluation":
                problem_data = load_evaluation_problem(args.problem_id, args.data_dir)
            else:
                problem_data = load_training_problem(args.problem_id, args.data_dir)
            
            result = inference.infer_single_problem(
                problem_data,
                train_sample_count=args.train_samples,
                verbose=True
            )
            
            print(f"\nFinal Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
            
        except Exception as e:
            print(f"Error processing problem {args.problem_id}: {e}")
    
    else:
        # Test multiple problems
        try:
            if args.dataset == "evaluation":
                problem_ids = list_evaluation_problems(args.data_dir)
            else:
                problem_ids = list_training_problems(args.data_dir)
            
            # Shuffle for random sampling
            random.shuffle(problem_ids)
            
            results = inference.evaluate_on_problems(
                problem_ids,
                data_dir=args.data_dir,
                dataset_type=args.dataset,
                max_problems=args.max_problems,
                train_sample_count=args.train_samples
            )
            
            # Save results if requested
            if args.output:
                print(f"\nSaving results to {args.output}")
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print("Results saved!")
        
        except Exception as e:
            print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
