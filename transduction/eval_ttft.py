"""
TTFT evaluation script for ARC transduction.

This script evaluates Test-Time Fine-Tuning (TTFT) with different base models
and inference approaches (Simple and AIRV).
"""

import sys
import os
from pathlib import Path
import argparse
import random
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transduction.eval import ARCEvaluator, ModelConfig, InferenceConfig
from transduction.inference.ttft import TTFTInference, TTFTWrapper, create_ttft_configs
from transduction.inference.airv import AIRVInference


def setup_ttft_evaluator(data_dir: str = ".") -> ARCEvaluator:
    """Set up evaluator with TTFT and comparison techniques."""
    evaluator = ARCEvaluator(data_dir=data_dir)
    
    # Set up default models
    evaluator.setup_default_models()
    
    # Register TTFT configurations
    ttft_configs = create_ttft_configs()
    for config in ttft_configs:
        # Register TTFT + Simple inference
        evaluator.register_inference_technique(InferenceConfig(
            name=f"ttft_{config['name']}",
            technique_class=TTFTInference,
            params=config['params'],
            description=f"TTFT + Simple: {config['description']}"
        ))
    
    # Register TTFT + AIRV combinations
    # We'll create a custom class for this
    class TTFTAIRVInference:
        """TTFT + AIRV combined inference technique."""
        
        def __init__(self, model_name: str, device: str = "auto", 
                     ttft_config: dict = None, airv_config: dict = None, **kwargs):
            self.ttft_config = ttft_config or {}
            self.airv_config = airv_config or {'num_augmentations': 4, 'include_original': True}
            
            # Create TTFT wrapper for AIRV
            self.ttft_wrapper = TTFTWrapper(AIRVInference, self.ttft_config)
            self.wrapped_inference = self.ttft_wrapper.create_wrapped_inference(
                model_name=model_name,
                device=device,
                **self.airv_config
            )
        
        def infer_single_problem(self, problem_data, train_sample_count=3, test_example_idx=0, verbose=False):
            return self.wrapped_inference.infer_single_problem(
                problem_data, train_sample_count, test_example_idx, verbose
            )
        
        def cleanup(self):
            if hasattr(self.wrapped_inference, 'cleanup'):
                self.wrapped_inference.cleanup()
    
    # Register TTFT + AIRV combinations
    for config in ttft_configs[:2]:  # Only register first 2 to avoid too many combinations
        evaluator.register_inference_technique(InferenceConfig(
            name=f"ttft_airv_{config['name']}",
            technique_class=TTFTAIRVInference,
            params={
                'ttft_config': config['params'],
                'airv_config': {'num_augmentations': 4, 'include_original': True}
            },
            description=f"TTFT + AIRV: {config['description']}"
        ))
    
    return evaluator


def compare_ttft_with_baselines(evaluator: ARCEvaluator, 
                               model_names: list = ['instruct'],
                               max_problems: int = 10,
                               verbose: bool = False):
    """
    Compare TTFT techniques with baseline methods.
    
    Args:
        evaluator: Configured evaluator
        model_names: Models to test
        max_problems: Number of problems to evaluate
        verbose: Verbose output
    """
    
    # Define techniques to compare
    baseline_techniques = ['standard', 'multi_sample_5']
    ttft_simple_techniques = ['ttft_ttft_light', 'ttft_ttft_standard']
    ttft_airv_techniques = ['ttft_airv_ttft_light', 'ttft_airv_ttft_standard']
    
    all_techniques = baseline_techniques + ttft_simple_techniques + ttft_airv_techniques
    
    print(f"Comparing TTFT with baselines:")
    print(f"  Models: {model_names}")
    print(f"  Baseline techniques: {baseline_techniques}")
    print(f"  TTFT + Simple techniques: {ttft_simple_techniques}")
    print(f"  TTFT + AIRV techniques: {ttft_airv_techniques}")
    print(f"  Max problems: {max_problems}")
    print("=" * 80)
    
    # Run evaluation
    results = evaluator.evaluate_all_combinations(
        model_names=model_names,
        technique_names=all_techniques,
        max_problems=max_problems,
        verbose=verbose
    )
    
    # Analyze results
    print(f"\n{'='*100}")
    print("TTFT vs BASELINE COMPARISON")
    print(f"{'='*100}")
    
    # Group by model
    model_results = {}
    for result in results:
        model_name = result.model_config.name
        if model_name not in model_results:
            model_results[model_name] = {}
        model_results[model_name][result.inference_config.name] = result
    
    # Print comparison for each model
    for model_name, techniques_results in model_results.items():
        print(f"\nModel: {model_name}")
        print("-" * 80)
        print(f"{'Technique':<25} {'Type':<15} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Avg Time':<10}")
        print("-" * 80)
        
        # Sort by accuracy descending
        sorted_techniques = sorted(
            techniques_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        for technique_name, result in sorted_techniques:
            # Determine technique type
            if technique_name in baseline_techniques:
                technique_type = "Baseline"
            elif technique_name.startswith('ttft_airv'):
                technique_type = "TTFT+AIRV"
            elif technique_name.startswith('ttft_'):
                technique_type = "TTFT+Simple"
            else:
                technique_type = "Other"
            
            print(f"{technique_name:<25} {technique_type:<15} {result.accuracy:<10.3f} "
                  f"{result.correct_predictions:<8} {result.total_problems:<8} "
                  f"{result.avg_inference_time:<10.1f}s")
        
        # Find best in each category
        baseline_results = {k: v for k, v in techniques_results.items() 
                          if k in baseline_techniques}
        ttft_simple_results = {k: v for k, v in techniques_results.items() 
                             if k.startswith('ttft_') and not k.startswith('ttft_airv')}
        ttft_airv_results = {k: v for k, v in techniques_results.items() 
                           if k.startswith('ttft_airv')}
        
        print(f"\nBest Results for {model_name}:")
        print("-" * 40)
        
        if baseline_results:
            best_baseline = max(baseline_results.values(), key=lambda x: x.accuracy)
            print(f"Best Baseline: {best_baseline.inference_config.name} "
                  f"({best_baseline.accuracy:.3f})")
        
        if ttft_simple_results:
            best_ttft_simple = max(ttft_simple_results.values(), key=lambda x: x.accuracy)
            print(f"Best TTFT+Simple: {best_ttft_simple.inference_config.name} "
                  f"({best_ttft_simple.accuracy:.3f})")
            
            if baseline_results:
                improvement = best_ttft_simple.accuracy - best_baseline.accuracy
                time_ratio = best_ttft_simple.avg_inference_time / best_baseline.avg_inference_time
                print(f"  vs Baseline: {improvement:+.3f} accuracy, {time_ratio:.1f}x time")
        
        if ttft_airv_results:
            best_ttft_airv = max(ttft_airv_results.values(), key=lambda x: x.accuracy)
            print(f"Best TTFT+AIRV: {best_ttft_airv.inference_config.name} "
                  f"({best_ttft_airv.accuracy:.3f})")
            
            if baseline_results:
                improvement = best_ttft_airv.accuracy - best_baseline.accuracy
                time_ratio = best_ttft_airv.avg_inference_time / best_baseline.avg_inference_time
                print(f"  vs Baseline: {improvement:+.3f} accuracy, {time_ratio:.1f}x time")
    
    return results


def compare_ttft_across_models(evaluator: ARCEvaluator,
                              max_problems: int = 10,
                              verbose: bool = False):
    """
    Compare TTFT effectiveness across different base models.
    
    Args:
        evaluator: Configured evaluator
        max_problems: Number of problems to evaluate
        verbose: Verbose output
    """
    
    # Get all available models
    model_names = list(evaluator.model_configs.keys())
    print(f"Comparing TTFT across models: {model_names}")
    
    # Use a subset of techniques for cross-model comparison
    techniques = ['standard', 'ttft_ttft_light', 'ttft_ttft_standard']
    
    print(f"Techniques: {techniques}")
    print(f"Max problems: {max_problems}")
    print("=" * 80)
    
    # Run evaluation
    results = evaluator.evaluate_all_combinations(
        model_names=model_names,
        technique_names=techniques,
        max_problems=max_problems,
        verbose=verbose
    )
    
    # Analyze results
    print(f"\n{'='*80}")
    print("TTFT EFFECTIVENESS ACROSS MODELS")
    print(f"{'='*80}")
    
    # Group by technique
    technique_results = {}
    for result in results:
        technique_name = result.inference_config.name
        if technique_name not in technique_results:
            technique_results[technique_name] = {}
        technique_results[technique_name][result.model_config.name] = result
    
    # Print comparison table
    print(f"\n{'Technique':<20} {'Model':<10} {'Accuracy':<10} {'Improvement':<12} {'Time Ratio':<10}")
    print("-" * 80)
    
    # For each model, compare TTFT vs baseline
    for model_name in model_names:
        baseline_result = None
        ttft_results = {}
        
        for technique_name, models_results in technique_results.items():
            if model_name in models_results:
                result = models_results[model_name]
                if technique_name == 'standard':
                    baseline_result = result
                elif technique_name.startswith('ttft_'):
                    ttft_results[technique_name] = result
        
        # Print baseline
        if baseline_result:
            print(f"{'standard':<20} {model_name:<10} {baseline_result.accuracy:<10.3f} "
                  f"{'(baseline)':<12} {'1.0x':<10}")
        
        # Print TTFT results with improvements
        for ttft_name, ttft_result in ttft_results.items():
            if baseline_result:
                improvement = ttft_result.accuracy - baseline_result.accuracy
                time_ratio = ttft_result.avg_inference_time / baseline_result.avg_inference_time
                print(f"{ttft_name:<20} {model_name:<10} {ttft_result.accuracy:<10.3f} "
                      f"{improvement:+.3f:<12} {time_ratio:.1f}x<10")
            else:
                print(f"{ttft_name:<20} {model_name:<10} {ttft_result.accuracy:<10.3f} "
                      f"{'N/A':<12} {'N/A':<10}")
        
        print("-" * 80)
    
    # Summary statistics
    print(f"\nSUMMARY:")
    
    # Calculate average improvements
    improvements = {
        'ttft_light': [],
        'ttft_standard': []
    }
    
    for model_name in model_names:
        baseline_acc = None
        ttft_accs = {}
        
        for technique_name, models_results in technique_results.items():
            if model_name in models_results:
                result = models_results[model_name]
                if technique_name == 'standard':
                    baseline_acc = result.accuracy
                elif technique_name == 'ttft_ttft_light':
                    ttft_accs['ttft_light'] = result.accuracy
                elif technique_name == 'ttft_ttft_standard':
                    ttft_accs['ttft_standard'] = result.accuracy
        
        if baseline_acc is not None:
            for ttft_type, ttft_acc in ttft_accs.items():
                improvement = ttft_acc - baseline_acc
                improvements[ttft_type].append(improvement)
    
    for ttft_type, improvement_list in improvements.items():
        if improvement_list:
            avg_improvement = sum(improvement_list) / len(improvement_list)
            print(f"Average {ttft_type} improvement: {avg_improvement:+.3f}")
    
    return results


def run_detailed_ttft_analysis(evaluator: ARCEvaluator,
                              model_name: str = 'instruct', 
                              max_problems: int = 3):
    """
    Run detailed analysis of TTFT technique on a few problems.
    
    Args:
        evaluator: Configured evaluator
        model_name: Model to analyze
        max_problems: Number of problems for detailed analysis
    """
    print(f"\nDetailed TTFT Analysis on {model_name}")
    print("=" * 60)
    
    # Run TTFT with verbose output
    results = evaluator.evaluate_all_combinations(
        model_names=[model_name],
        technique_names=['ttft_ttft_standard'],
        max_problems=max_problems,
        verbose=True  # This will show detailed TTFT steps
    )
    
    if results:
        result = results[0]
        print(f"\nDetailed Analysis Summary:")
        print(f"Problems analyzed: {result.total_problems}")
        print(f"Correct predictions: {result.correct_predictions}")
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Average inference time: {result.avg_inference_time:.1f}s")
        
        # Analyze TTFT-specific metrics
        ttft_stats = {
            'total_training_examples': 0,
            'successful_fine_tuning': 0,
            'failed_fine_tuning': 0,
            'average_epochs': 0
        }
        
        for problem_result in result.problem_results:
            if 'ttft_epochs' in problem_result:
                ttft_stats['average_epochs'] += problem_result['ttft_epochs']
                ttft_stats['successful_fine_tuning'] += 1
            else:
                ttft_stats['failed_fine_tuning'] += 1
        
        if result.total_problems > 0:
            ttft_stats['average_epochs'] /= result.total_problems
        
        print(f"\nTTFT Statistics:")
        print(f"Average epochs per problem: {ttft_stats['average_epochs']:.1f}")
        print(f"Successful fine-tuning: {ttft_stats['successful_fine_tuning']}")
        print(f"Failed fine-tuning: {ttft_stats['failed_fine_tuning']}")


def main():
    """Main function for TTFT evaluation."""
    parser = argparse.ArgumentParser(description='TTFT Evaluation for ARC Transduction')
    parser.add_argument('--data_dir', type=str, default=".", help='Data directory')
    parser.add_argument('--models', type=str, nargs='+', default=['instruct'], 
                       help='Models to evaluate')
    parser.add_argument('--max_problems', type=int, default=10,
                       help='Maximum problems to evaluate')
    parser.add_argument('--cross_model', action='store_true',
                       help='Compare TTFT effectiveness across different models')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Run detailed analysis on a few problems')
    parser.add_argument('--detailed_problems', type=int, default=3,
                       help='Number of problems for detailed analysis')
    parser.add_argument('--output', type=str, default="ttft_results.json",
                       help='Output file for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass
    
    print("🔧 TTFT (Test-Time Fine-Tuning) Evaluation")
    print("=" * 80)
    
    # Set up evaluator
    evaluator = setup_ttft_evaluator(args.data_dir)
    
    start_time = time.time()
    
    if args.cross_model:
        # Compare TTFT across different models
        results = compare_ttft_across_models(
            evaluator,
            max_problems=args.max_problems,
            verbose=args.verbose
        )
    else:
        # Run comparison with baselines
        results = compare_ttft_with_baselines(
            evaluator,
            model_names=args.models,
            max_problems=args.max_problems,
            verbose=args.verbose
        )
    
    # Save results
    evaluator.save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    
    # Run detailed analysis if requested
    if args.detailed_analysis:
        run_detailed_ttft_analysis(
            evaluator,
            model_name=args.models[0],
            max_problems=args.detailed_problems
        )
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.1f}s")
    print("TTFT evaluation completed! 🚀")


if __name__ == "__main__":
    main()
