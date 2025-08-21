"""
AIRV evaluation script for ARC transduction.

This script demonstrates how to use the AIRV (Augment, Infer, Revert and Vote)
technique for ARC problem solving and compares it with other methods.
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
from transduction.inference.airv import AIRVInference, create_airv_configs


def setup_airv_evaluator(data_dir: str = ".") -> ARCEvaluator:
    """Set up evaluator with AIRV and comparison techniques."""
    evaluator = ARCEvaluator(data_dir=data_dir)
    
    # Set up default models
    evaluator.setup_default_models()
    
    # Register AIRV configurations
    airv_configs = create_airv_configs()
    for config in airv_configs:
        evaluator.register_inference_technique(InferenceConfig(
            name=config['name'],
            technique_class=AIRVInference,
            params=config['params'],
            description=config['description']
        ))
    
    return evaluator


def compare_airv_with_baselines(evaluator: ARCEvaluator, 
                               model_names: list = ['instruct'],
                               max_problems: int = 20,
                               verbose: bool = False):
    """
    Compare AIRV techniques with baseline methods.
    
    Args:
        evaluator: Configured evaluator
        model_names: Models to test
        max_problems: Number of problems to evaluate
        verbose: Verbose output
    """
    
    # Define techniques to compare
    baseline_techniques = ['standard', 'multi_sample_5']
    airv_techniques = ['airv_light', 'airv_standard']
    all_techniques = baseline_techniques + airv_techniques
    
    print(f"Comparing AIRV with baselines:")
    print(f"  Models: {model_names}")
    print(f"  Baseline techniques: {baseline_techniques}")
    print(f"  AIRV techniques: {airv_techniques}")
    print(f"  Max problems: {max_problems}")
    print("=" * 60)
    
    # Run evaluation
    results = evaluator.evaluate_all_combinations(
        model_names=model_names,
        technique_names=all_techniques,
        max_problems=max_problems,
        verbose=verbose
    )
    
    # Analyze results
    print(f"\n{'='*80}")
    print("AIRV vs BASELINE COMPARISON")
    print(f"{'='*80}")
    
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
        print("-" * 40)
        print(f"{'Technique':<20} {'Accuracy':<10} {'Avg Time':<10} {'Problems':<10}")
        print("-" * 40)
        
        # Sort by accuracy descending
        sorted_techniques = sorted(
            techniques_results.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        for technique_name, result in sorted_techniques:
            technique_type = "AIRV" if technique_name.startswith('airv') else "Baseline"
            print(f"{technique_name:<20} {result.accuracy:<10.3f} "
                  f"{result.avg_inference_time:<10.1f} {result.total_problems:<10}")
        
        # Find best baseline and best AIRV
        baseline_results = {k: v for k, v in techniques_results.items() 
                          if not k.startswith('airv')}
        airv_results = {k: v for k, v in techniques_results.items() 
                       if k.startswith('airv')}
        
        if baseline_results and airv_results:
            best_baseline = max(baseline_results.values(), key=lambda x: x.accuracy)
            best_airv = max(airv_results.values(), key=lambda x: x.accuracy)
            
            improvement = best_airv.accuracy - best_baseline.accuracy
            time_ratio = best_airv.avg_inference_time / best_baseline.avg_inference_time
            
            print(f"\nBest Baseline: {best_baseline.inference_config.name} "
                  f"({best_baseline.accuracy:.3f})")
            print(f"Best AIRV: {best_airv.inference_config.name} "
                  f"({best_airv.accuracy:.3f})")
            print(f"Improvement: {improvement:+.3f} accuracy")
            print(f"Time cost: {time_ratio:.1f}x slower")
    
    return results


def run_detailed_airv_analysis(evaluator: ARCEvaluator,
                              model_name: str = 'instruct', 
                              max_problems: int = 5):
    """
    Run detailed analysis of AIRV technique on a few problems.
    
    Args:
        evaluator: Configured evaluator
        model_name: Model to analyze
        max_problems: Number of problems for detailed analysis
    """
    print(f"\nDetailed AIRV Analysis on {model_name}")
    print("=" * 50)
    
    # Run AIRV with verbose output
    results = evaluator.evaluate_all_combinations(
        model_names=[model_name],
        technique_names=['airv_standard'],
        max_problems=max_problems,
        verbose=True  # This will show detailed AIRV steps
    )
    
    if results:
        result = results[0]
        print(f"\nDetailed Analysis Summary:")
        print(f"Problems analyzed: {result.total_problems}")
        print(f"Correct predictions: {result.correct_predictions}")
        print(f"Accuracy: {result.accuracy:.3f}")
        print(f"Average inference time: {result.avg_inference_time:.1f}s")
        
        # Analyze AIRV-specific metrics
        airv_stats = {
            'total_versions': 0,
            'successful_reversions': 0,
            'failed_reversions': 0,
            'vote_ties': 0
        }
        
        for problem_result in result.problem_results:
            if 'num_versions' in problem_result:
                airv_stats['total_versions'] += problem_result['num_versions']
            
            if 'reversion_info' in problem_result:
                for reversion in problem_result['reversion_info']:
                    if reversion['reversion_success']:
                        airv_stats['successful_reversions'] += 1
                    else:
                        airv_stats['failed_reversions'] += 1
        
        print(f"\nAIRV Statistics:")
        print(f"Average versions per problem: {airv_stats['total_versions'] / result.total_problems:.1f}")
        print(f"Successful reversions: {airv_stats['successful_reversions']}")
        print(f"Failed reversions: {airv_stats['failed_reversions']}")
        reversion_rate = airv_stats['successful_reversions'] / (airv_stats['successful_reversions'] + airv_stats['failed_reversions'])
        print(f"Reversion success rate: {reversion_rate:.3f}")


def main():
    """Main function for AIRV evaluation."""
    parser = argparse.ArgumentParser(description='AIRV Evaluation for ARC Transduction')
    parser.add_argument('--data_dir', type=str, default=".", help='Data directory')
    parser.add_argument('--models', type=str, nargs='+', default=['instruct'], 
                       help='Models to evaluate')
    parser.add_argument('--max_problems', type=int, default=20,
                       help='Maximum problems to evaluate')
    parser.add_argument('--detailed_analysis', action='store_true',
                       help='Run detailed analysis on a few problems')
    parser.add_argument('--detailed_problems', type=int, default=5,
                       help='Number of problems for detailed analysis')
    parser.add_argument('--output', type=str, default="airv_results.json",
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
    
    print("🎯 AIRV (Augment, Infer, Revert and Vote) Evaluation")
    print("=" * 60)
    
    # Set up evaluator
    evaluator = setup_airv_evaluator(args.data_dir)
    
    # Run comparison with baselines
    start_time = time.time()
    results = compare_airv_with_baselines(
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
        run_detailed_airv_analysis(
            evaluator,
            model_name=args.models[0],
            max_problems=args.detailed_problems
        )
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.1f}s")
    print("AIRV evaluation completed! 🚀")


if __name__ == "__main__":
    main()
