"""
Comprehensive evaluation system for ARC transduction models.

This module provides a flexible framework for evaluating different model types
(instruct, SFT, RL) and inference techniques on ARC problems.
"""

import json
import os
import sys
import argparse
import random
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from loader import load_evaluation_problem, load_training_problem, list_evaluation_problems, list_training_problems
from transduction.inference.inference import ARCTransductionInference

# HuggingFace imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers and/or torch not available. Please install with: pip install transformers torch")
    HF_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    name: str                    # Human-readable name
    model_type: str             # "instruct", "sft", "rl"
    model_path: str             # Path to model or HF model name
    base_model: Optional[str] = None  # Base model for LoRA adapters
    description: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for an inference technique."""
    name: str                    # Human-readable name
    technique_class: type       # Class implementing the inference technique
    params: Dict[str, Any]      # Parameters for the technique
    description: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from evaluating a single model-technique combination."""
    model_config: ModelConfig
    inference_config: InferenceConfig
    problem_results: List[Dict[str, Any]]
    total_problems: int
    correct_predictions: int
    accuracy: float
    avg_inference_time: float
    total_time: float
    metadata: Dict[str, Any]


class InferenceTechnique(ABC):
    """Abstract base class for inference techniques."""
    
    @abstractmethod
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        """Initialize the inference technique."""
        pass
    
    @abstractmethod
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        """Perform inference on a single problem."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class StandardInference(InferenceTechnique):
    """Standard inference technique using the existing ARCTransductionInference."""
    
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.inference = ARCTransductionInference(model_name=model_name, device=device)
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        return self.inference.infer_single_problem(
            problem_data, train_sample_count, test_example_idx, verbose
        )
    
    def cleanup(self):
        # Clean up GPU memory
        if hasattr(self.inference, 'model'):
            del self.inference.model
        if hasattr(self.inference, 'tokenizer'):
            del self.inference.tokenizer
        if HF_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


class MultiSampleInference(InferenceTechnique):
    """Inference technique that generates multiple samples and picks the best."""
    
    def __init__(self, model_name: str, device: str = "auto", num_samples: int = 5, **kwargs):
        self.inference = ARCTransductionInference(model_name=model_name, device=device)
        self.num_samples = num_samples
        
        # Modify generation config for sampling
        self.inference.generation_config.temperature = 0.8
        self.inference.generation_config.do_sample = True
        self.inference.generation_config.top_p = 0.9
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        # Generate prompt once
        prompt = self.inference.format_problem_for_inference(
            problem_data, train_sample_count, test_example_idx
        )
        
        # Generate multiple samples
        samples = []
        for i in range(self.num_samples):
            response = self.inference.generate_response(prompt)
            predicted_grid = self.inference.parse_grid_response(response)
            samples.append({
                'response': response,
                'predicted_grid': predicted_grid,
                'sample_idx': i
            })
        
        # Get ground truth
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        
        # Find the best sample (first correct one, or first parseable one)
        best_sample = None
        for sample in samples:
            if sample['predicted_grid'] is not None:
                is_correct = self.inference.evaluate_prediction(sample['predicted_grid'], ground_truth)
                if is_correct:
                    best_sample = sample
                    break
                elif best_sample is None:  # First parseable sample
                    best_sample = sample
        
        # Fallback to first sample if none are parseable
        if best_sample is None:
            best_sample = samples[0]
        
        # Evaluate best sample
        is_correct = self.inference.evaluate_prediction(best_sample['predicted_grid'], ground_truth)
        
        result = {
            'prompt': prompt,
            'response': best_sample['response'],
            'predicted_grid': best_sample['predicted_grid'],
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'train_sample_count': train_sample_count,
            'test_example_idx': test_example_idx,
            'all_samples': samples,
            'best_sample_idx': best_sample['sample_idx'],
            'num_samples': self.num_samples
        }
        
        if verbose:
            print(f"Generated {self.num_samples} samples, selected sample {best_sample['sample_idx']}")
            print(f"GROUND TRUTH:")
            for row in ground_truth:
                print(';'.join(map(str, row)))
            print()
            
            print(f"PREDICTED:")
            if best_sample['predicted_grid']:
                for row in best_sample['predicted_grid']:
                    print(';'.join(map(str, row)))
            else:
                print("Failed to parse prediction")
            print()
            
            print(f"CORRECT: {is_correct}")
            print("=" * 50)
        
        return result
    
    def cleanup(self):
        if hasattr(self.inference, 'model'):
            del self.inference.model
        if hasattr(self.inference, 'tokenizer'):
            del self.inference.tokenizer
        if HF_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


class ARCEvaluator:
    """Main evaluator class for ARC transduction models."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.model_configs = {}
        self.inference_configs = {}
        self._register_default_techniques()
    
    def _register_default_techniques(self):
        """Register default inference techniques."""
        self.register_inference_technique(InferenceConfig(
            name="standard",
            technique_class=StandardInference,
            params={},
            description="Standard single-sample inference"
        ))
        
        self.register_inference_technique(InferenceConfig(
            name="multi_sample_5",
            technique_class=MultiSampleInference,
            params={"num_samples": 5},
            description="Multi-sample inference with 5 samples"
        ))
        
        self.register_inference_technique(InferenceConfig(
            name="multi_sample_10",
            technique_class=MultiSampleInference,
            params={"num_samples": 10},
            description="Multi-sample inference with 10 samples"
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        self.model_configs[config.name] = config
    
    def register_inference_technique(self, config: InferenceConfig):
        """Register an inference technique."""
        self.inference_configs[config.name] = config
    
    def setup_default_models(self, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """Set up default model configurations."""
        # Instruct model
        self.register_model(ModelConfig(
            name="instruct",
            model_type="instruct",
            model_path=base_model,
            description=f"Base instruct model: {base_model}"
        ))
        
        # SFT model (assumes standard path structure)
        sft_path = "qwen2.5_0.5b_arc_transduction_sft/final"
        if Path(sft_path).exists():
            self.register_model(ModelConfig(
                name="sft",
                model_type="sft",
                model_path=sft_path,
                base_model=base_model,
                description=f"SFT model based on {base_model}"
            ))
        
        # RL model (assumes standard path structure)
        rl_path = "qwen2.5_0.5b_arc_transduction_rl/final"
        if Path(rl_path).exists():
            self.register_model(ModelConfig(
                name="rl",
                model_type="rl",
                model_path=rl_path,
                base_model=base_model,
                description=f"RL model based on {base_model}"
            ))
    
    def evaluate_single_combination(self, 
                                  model_config: ModelConfig,
                                  inference_config: InferenceConfig,
                                  problem_ids: List[str],
                                  dataset_type: str = "evaluation",
                                  max_problems: Optional[int] = None,
                                  train_sample_count: int = 3,
                                  verbose: bool = False) -> EvaluationResult:
        """Evaluate a single model-technique combination."""
        
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_config.name} + {inference_config.name}")
        print(f"Model: {model_config.model_path}")
        print(f"Problems: {len(problem_ids)} from {dataset_type} set")
        print(f"{'='*60}")
        
        # Initialize inference technique
        technique = inference_config.technique_class(
            model_name=model_config.model_path,
            device="auto",
            **inference_config.params
        )
        
        results = []
        correct_count = 0
        total_inference_time = 0
        start_time = time.time()
        
        try:
            for i, problem_id in enumerate(problem_ids):
                if verbose:
                    print(f"\nProblem {i+1}/{len(problem_ids)}: {problem_id}")
                
                try:
                    # Load problem
                    if dataset_type == "evaluation":
                        problem_data = load_evaluation_problem(problem_id, self.data_dir)
                    else:
                        problem_data = load_training_problem(problem_id, self.data_dir)
                    
                    # Perform inference
                    inference_start = time.time()
                    result = technique.infer_single_problem(
                        problem_data, 
                        train_sample_count=train_sample_count,
                        verbose=verbose
                    )
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time
                    
                    result['problem_id'] = problem_id
                    result['inference_time'] = inference_time
                    results.append(result)
                    
                    if result['is_correct']:
                        correct_count += 1
                        if not verbose:
                            print(f"✓ {problem_id}")
                    else:
                        if not verbose:
                            print(f"✗ {problem_id}")
                    
                except Exception as e:
                    print(f"Error processing problem {problem_id}: {e}")
                    continue
            
        finally:
            # Clean up resources
            technique.cleanup()
        
        total_time = time.time() - start_time
        accuracy = correct_count / len(results) if results else 0
        avg_inference_time = total_inference_time / len(results) if results else 0
        
        evaluation_result = EvaluationResult(
            model_config=model_config,
            inference_config=inference_config,
            problem_results=results,
            total_problems=len(results),
            correct_predictions=correct_count,
            accuracy=accuracy,
            avg_inference_time=avg_inference_time,
            total_time=total_time,
            metadata={
                'dataset_type': dataset_type,
                'train_sample_count': train_sample_count,
                'max_problems': max_problems
            }
        )
        
        print(f"\nResults for {model_config.name} + {inference_config.name}:")
        print(f"  Total problems: {len(results)}")
        print(f"  Correct: {correct_count}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Avg inference time: {avg_inference_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        
        return evaluation_result
    
    def evaluate_all_combinations(self,
                                model_names: Optional[List[str]] = None,
                                technique_names: Optional[List[str]] = None,
                                dataset_type: str = "evaluation",
                                max_problems: Optional[int] = None,
                                train_sample_count: int = 3,
                                output_file: Optional[str] = None,
                                verbose: bool = False) -> List[EvaluationResult]:
        """Evaluate all specified model-technique combinations."""
        
        # Get problem IDs
        if dataset_type == "evaluation":
            problem_ids = list_evaluation_problems(self.data_dir)
        else:
            problem_ids = list_training_problems(self.data_dir)
        
        # Shuffle for random sampling
        random.shuffle(problem_ids)
        
        # Determine which models and techniques to evaluate
        if model_names is None:
            model_names = list(self.model_configs.keys())
        if technique_names is None:
            technique_names = list(self.inference_configs.keys())
        
        print(f"Evaluating {len(model_names)} models × {len(technique_names)} techniques")
        print(f"Models: {model_names}")
        print(f"Techniques: {technique_names}")
        
        all_results = []
        
        for model_name in model_names:
            if model_name not in self.model_configs:
                print(f"Warning: Model '{model_name}' not found, skipping")
                continue
                
            for technique_name in technique_names:
                if technique_name not in self.inference_configs:
                    print(f"Warning: Technique '{technique_name}' not found, skipping")
                    continue
                
                try:
                    result = self.evaluate_single_combination(
                        self.model_configs[model_name],
                        self.inference_configs[technique_name],
                        problem_ids,
                        dataset_type=dataset_type,
                        max_problems=max_problems,
                        train_sample_count=train_sample_count,
                        verbose=verbose
                    )
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} + {technique_name}: {e}")
                    continue
        
        # Print summary
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'Technique':<15} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Avg Time':<10}")
        print("-" * 80)
        
        for result in all_results:
            print(f"{result.model_config.name:<15} "
                  f"{result.inference_config.name:<15} "
                  f"{result.accuracy:<10.3f} "
                  f"{result.correct_predictions:<8} "
                  f"{result.total_problems:<8} "
                  f"{result.avg_inference_time:<10.2f}s")
        
        # Save results if requested
        if output_file:
            print(f"\nSaving detailed results to {output_file}")
            self.save_results(all_results, output_file)
        
        return all_results
    
    def save_results(self, results: List[EvaluationResult], output_file: str):
        """Save evaluation results to JSON file."""
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'model_config': {
                    'name': result.model_config.name,
                    'model_type': result.model_config.model_type,
                    'model_path': result.model_config.model_path,
                    'base_model': result.model_config.base_model,
                    'description': result.model_config.description
                },
                'inference_config': {
                    'name': result.inference_config.name,
                    'params': result.inference_config.params,
                    'description': result.inference_config.description
                },
                'problem_results': result.problem_results,
                'total_problems': result.total_problems,
                'correct_predictions': result.correct_predictions,
                'accuracy': result.accuracy,
                'avg_inference_time': result.avg_inference_time,
                'total_time': result.total_time,
                'metadata': result.metadata
            }
            serializable_results.append(serializable_result)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    """Main function for running comprehensive ARC evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive ARC Transduction Evaluation')
    
    # Model configuration
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help='Base model for instruct/SFT/RL variants')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to evaluate (default: all available)')
    parser.add_argument('--sft_path', type=str, default="qwen2.5_0.5b_arc_transduction_sft/final",
                       help='Path to SFT model')
    parser.add_argument('--rl_path', type=str, default="qwen2.5_0.5b_arc_transduction_rl/final",
                       help='Path to RL model')
    
    # Inference techniques
    parser.add_argument('--techniques', type=str, nargs='+', default=None,
                       help='Inference techniques to use (default: all available)')
    
    # Evaluation settings
    parser.add_argument('--data_dir', type=str, default=".",
                       help='Directory containing ARC data')
    parser.add_argument('--dataset', type=str, choices=['evaluation', 'training'], 
                       default='evaluation',
                       help='Dataset to evaluate on')
    parser.add_argument('--max_problems', type=int, default=50,
                       help='Maximum number of problems to evaluate')
    parser.add_argument('--train_samples', type=int, default=3,
                       help='Number of training examples to use per problem')
    
    # Output and misc
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save detailed results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    if HF_AVAILABLE:
        torch.manual_seed(args.seed)
    
    # Initialize evaluator
    evaluator = ARCEvaluator(data_dir=args.data_dir)
    
    # Set up models
    evaluator.setup_default_models(base_model=args.base_model)
    
    # Override default paths if provided
    if args.sft_path != "qwen2.5_0.5b_arc_transduction_sft/final" or Path(args.sft_path).exists():
        evaluator.register_model(ModelConfig(
            name="sft",
            model_type="sft", 
            model_path=args.sft_path,
            base_model=args.base_model,
            description=f"Custom SFT model: {args.sft_path}"
        ))
    
    if args.rl_path != "qwen2.5_0.5b_arc_transduction_rl/final" or Path(args.rl_path).exists():
        evaluator.register_model(ModelConfig(
            name="rl",
            model_type="rl",
            model_path=args.rl_path, 
            base_model=args.base_model,
            description=f"Custom RL model: {args.rl_path}"
        ))
    
    # Run evaluation
    results = evaluator.evaluate_all_combinations(
        model_names=args.models,
        technique_names=args.techniques,
        dataset_type=args.dataset,
        max_problems=args.max_problems,
        train_sample_count=args.train_samples,
        output_file=args.output,
        verbose=args.verbose
    )
    
    print(f"\nEvaluation completed! Evaluated {len(results)} model-technique combinations.")


if __name__ == "__main__":
    main()
