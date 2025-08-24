#!/usr/bin/env python3
"""
Comprehensive AIRV (Augment, Infer, Revert, Vote) Testing Framework

This script tests thousands of augmentation/reversion combinations to identify and fix
issues with the AIRV system. It performs systematic validation of the augmentation
and deaugmentation pipeline.

Features:
- Tests all possible augmentation combinations
- Validates perfect round-trip augment->deaugment operations
- Identifies problematic combinations and edge cases
- Provides detailed statistics and error reporting
- Tests with real ARC problems from training/evaluation datasets
- Generates comprehensive test reports
"""

import json
import random
import itertools
import traceback
from typing import Dict, List, Any, Tuple, Optional, Set
from copy import deepcopy
from pathlib import Path
import sys
import os
from collections import defaultdict, Counter
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augment import (
    apply_random_augmentations, 
    get_available_augmentations, 
    apply_augmentation_to_problem,
    track_pixel_transformations
)
from deaugment import (
    apply_full_deaugmentation, 
    apply_pixel_level_deaugmentation,
    create_augmentation_metadata,
    get_deaugmentation_functions
)
from loader import (
    load_evaluation_problem, 
    load_training_problem,
    list_evaluation_problems, 
    list_training_problems
)


class AIRVTestResult:
    """Container for individual test results."""
    
    def __init__(self, test_id: str, problem_id: str, augmentations: List[str]):
        self.test_id = test_id
        self.problem_id = problem_id
        self.augmentations = augmentations
        self.success = False
        self.error_message = None
        self.error_type = None
        self.original_size = None
        self.augmented_size = None
        self.reverted_size = None
        self.perfect_reversion = False
        self.metadata = {}
        self.execution_time = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_id': self.test_id,
            'problem_id': self.problem_id,
            'augmentations': self.augmentations,
            'success': self.success,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'original_size': self.original_size,
            'augmented_size': self.augmented_size,
            'reverted_size': self.reverted_size,
            'perfect_reversion': self.perfect_reversion,
            'metadata': self.metadata,
            'execution_time': self.execution_time
        }


class AIRVComprehensiveTester:
    """Comprehensive testing framework for AIRV augmentation/reversion pipeline."""
    
    def __init__(self, data_dir: str = ".", max_problems: int = 50, verbose: bool = True):
        """
        Initialize the comprehensive tester.
        
        Args:
            data_dir: Directory containing ARC datasets
            max_problems: Maximum number of problems to test with
            verbose: Whether to print detailed progress
        """
        self.data_dir = data_dir
        self.max_problems = max_problems
        self.verbose = verbose
        
        # Load available problems
        try:
            self.eval_problems = list_evaluation_problems(data_dir)[:max_problems//2]
            self.train_problems = list_training_problems(data_dir)[:max_problems//2]
            self.all_problems = self.eval_problems + self.train_problems
        except Exception as e:
            print(f"Warning: Could not load all problems: {e}")
            self.eval_problems = []
            self.train_problems = []
            self.all_problems = []
        
        # Get available augmentations
        self.augmentation_funcs = get_available_augmentations()
        self.deaugmentation_funcs = get_deaugmentation_functions()
        
        # Test results storage
        self.results: List[AIRVTestResult] = []
        self.stats = defaultdict(int)
        self.error_patterns = defaultdict(list)
        
        if self.verbose:
            print(f"Initialized AIRV tester with {len(self.all_problems)} problems")
            print(f"Available augmentations: {list(self.augmentation_funcs.keys())}")
    
    def generate_augmentation_combinations(self, max_augs: int = 4) -> List[List[str]]:
        """
        Generate all possible combinations of augmentations up to max_augs length.
        
        Args:
            max_augs: Maximum number of augmentations per combination
            
        Returns:
            List of augmentation combinations
        """
        aug_names = list(self.augmentation_funcs.keys())
        combinations = []
        
        # Single augmentations
        for aug in aug_names:
            combinations.append([aug])
        
        # Pairs
        for combo in itertools.combinations(aug_names, 2):
            combinations.append(list(combo))
        
        # Triples
        for combo in itertools.combinations(aug_names, 3):
            combinations.append(list(combo))
        
        # Quadruples (if requested)
        if max_augs >= 4:
            for combo in itertools.combinations(aug_names, 4):
                combinations.append(list(combo))
        
        # Add some permutations to test order dependency
        popular_combos = [
            ['rotate_90', 'flip_horizontal'],
            ['color_permutation', 'rotate_180'],
            ['upscale', 'rotate_90'],
            ['flip_vertical', 'upscale'],
            ['rotate_270', 'color_permutation', 'flip_horizontal']
        ]
        
        for combo in popular_combos:
            # Add different orders
            for perm in itertools.permutations(combo):
                if list(perm) not in combinations:
                    combinations.append(list(perm))
        
        return combinations
    
    def test_single_combination(self, problem_id: str, augmentations: List[str], 
                              test_id: str) -> AIRVTestResult:
        """
        Test a single augmentation combination on a specific problem.
        
        Args:
            problem_id: ID of the problem to test
            augmentations: List of augmentations to apply
            test_id: Unique test identifier
            
        Returns:
            Test result object
        """
        result = AIRVTestResult(test_id, problem_id, augmentations)
        start_time = time.time()
        
        try:
            # Load the problem
            if problem_id in self.eval_problems:
                problem_data = load_evaluation_problem(problem_id, self.data_dir)
            else:
                problem_data = load_training_problem(problem_id, self.data_dir)
            
            if not problem_data:
                result.error_message = "Failed to load problem"
                result.error_type = "LOAD_ERROR"
                return result
            
            # Get original size from first training example
            if problem_data.get('train') and problem_data['train'][0].get('input'):
                original_grid = problem_data['train'][0]['input']
                result.original_size = (len(original_grid), len(original_grid[0]))
            
            # Apply augmentations manually (controlled sequence)
            augmented_problem = self._apply_controlled_augmentations(
                problem_data, augmentations
            )
            
            if not augmented_problem[0]:  # augmented_problem is (problem, applied_augs, params)
                result.error_message = "Failed to apply augmentations"
                result.error_type = "AUGMENT_ERROR"
                return result
            
            augmented_prob, applied_augs, aug_params = augmented_problem
            
            # Get augmented size
            if augmented_prob.get('train') and augmented_prob['train'][0].get('input'):
                aug_grid = augmented_prob['train'][0]['input']
                result.augmented_size = (len(aug_grid), len(aug_grid[0]))
            
            # Create comprehensive metadata
            metadata = create_augmentation_metadata(problem_data, applied_augs)
            metadata['augmentation_params'] = aug_params
            
            # Test deaugmentation
            reverted_problem = apply_full_deaugmentation(
                augmented_prob, applied_augs, metadata
            )
            
            # Get reverted size
            if reverted_problem.get('train') and reverted_problem['train'][0].get('input'):
                rev_grid = reverted_problem['train'][0]['input']
                result.reverted_size = (len(rev_grid), len(rev_grid[0]))
            
            # Check if reversion is perfect
            result.perfect_reversion = self._compare_problems(problem_data, reverted_problem)
            result.success = True
            result.metadata = {
                'applied_augmentations': applied_augs,
                'augmentation_params': aug_params,
                'metadata_keys': list(metadata.keys())
            }
            
        except Exception as e:
            result.error_message = str(e)
            result.error_type = type(e).__name__
            result.metadata['traceback'] = traceback.format_exc()
            
        result.execution_time = time.time() - start_time
        return result
    
    def _apply_controlled_augmentations(self, problem: Dict[str, Any], 
                                      augmentations: List[str]) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
        """
        Apply a controlled sequence of augmentations with proper parameter tracking.
        
        Args:
            problem: Original problem data
            augmentations: Sequence of augmentations to apply
            
        Returns:
            Tuple of (augmented_problem, applied_augmentations, parameters)
        """
        try:
            augmented_problem = deepcopy(problem)
            applied_augs = []
            aug_params = {}
            
            for aug_name in augmentations:
                if aug_name not in self.augmentation_funcs:
                    continue
                
                aug_func = self.augmentation_funcs[aug_name]
                
                # Handle special cases
                if aug_name == 'color_permutation':
                    # Generate consistent color map
                    colors = list(range(10))
                    shuffled_colors = colors.copy()
                    random.shuffle(shuffled_colors)
                    color_map = dict(zip(colors, shuffled_colors))
                    
                    aug_params[aug_name] = {'color_map': color_map}
                    augmented_problem = apply_augmentation_to_problem(
                        augmented_problem, aug_func, color_map=color_map
                    )
                    
                elif aug_name == 'upscale':
                    # Get current size for consistent upscaling
                    if augmented_problem.get('train') and augmented_problem['train'][0].get('input'):
                        current_grid = augmented_problem['train'][0]['input']
                        current_h, current_w = len(current_grid), len(current_grid[0])
                        
                        # Choose reasonable target size
                        target_h = current_h + random.randint(1, min(5, 30 - current_h))
                        target_w = current_w + random.randint(1, min(5, 30 - current_w))
                        target_size = (target_h, target_w)
                        
                        upscale_name = f"upscale_{target_h}x{target_w}"
                        aug_params[upscale_name] = {
                            'original_size': (current_h, current_w),
                            'target_size': target_size
                        }
                        
                        augmented_problem = apply_augmentation_to_problem(
                            augmented_problem, aug_func, target_size=target_size
                        )
                        applied_augs.append(upscale_name)
                        continue
                
                else:
                    # Simple augmentation
                    augmented_problem = apply_augmentation_to_problem(
                        augmented_problem, aug_func
                    )
                
                applied_augs.append(aug_name)
            
            return augmented_problem, applied_augs, aug_params
            
        except Exception as e:
            return None, [], {}
    
    def _compare_problems(self, original: Dict[str, Any], reverted: Dict[str, Any]) -> bool:
        """
        Compare two problems to check if they are identical.
        
        Args:
            original: Original problem data
            reverted: Reverted problem data
            
        Returns:
            True if problems are identical
        """
        try:
            # Compare training examples
            if len(original.get('train', [])) != len(reverted.get('train', [])):
                return False
            
            for orig_ex, rev_ex in zip(original.get('train', []), reverted.get('train', [])):
                if orig_ex.get('input') != rev_ex.get('input'):
                    return False
                if orig_ex.get('output') != rev_ex.get('output'):
                    return False
            
            # Compare test examples
            if len(original.get('test', [])) != len(reverted.get('test', [])):
                return False
            
            for orig_ex, rev_ex in zip(original.get('test', []), reverted.get('test', [])):
                if orig_ex.get('input') != rev_ex.get('input'):
                    return False
                # Note: test outputs might not exist in all cases
                if 'output' in orig_ex and 'output' in rev_ex:
                    if orig_ex['output'] != rev_ex['output']:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def run_comprehensive_tests(self, max_combinations: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive tests on thousands of combinations.
        
        Args:
            max_combinations: Maximum number of combinations to test
            
        Returns:
            Comprehensive test results
        """
        if self.verbose:
            print("="*80)
            print("STARTING COMPREHENSIVE AIRV TESTING")
            print("="*80)
        
        # Generate all augmentation combinations
        all_combinations = self.generate_augmentation_combinations()
        
        # Limit combinations if needed
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        total_tests = len(all_combinations) * len(self.all_problems)
        
        if self.verbose:
            print(f"Generated {len(all_combinations)} augmentation combinations")
            print(f"Testing with {len(self.all_problems)} problems")
            print(f"Total tests to run: {total_tests}")
            print()
        
        test_count = 0
        start_time = time.time()
        
        # Run tests
        for combo_idx, combo in enumerate(all_combinations):
            for prob_idx, problem_id in enumerate(self.all_problems):
                test_id = f"test_{combo_idx:04d}_{prob_idx:03d}"
                
                result = self.test_single_combination(problem_id, combo, test_id)
                self.results.append(result)
                
                # Update statistics
                self.stats['total_tests'] += 1
                if result.success:
                    self.stats['successful_tests'] += 1
                    if result.perfect_reversion:
                        self.stats['perfect_reversions'] += 1
                else:
                    self.stats['failed_tests'] += 1
                    self.error_patterns[result.error_type].append(result)
                
                test_count += 1
                
                # Progress reporting
                if self.verbose and test_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = test_count / elapsed
                    eta = (total_tests - test_count) / rate if rate > 0 else 0
                    
                    print(f"Progress: {test_count}/{total_tests} "
                          f"({test_count/total_tests*100:.1f}%) "
                          f"Rate: {rate:.1f} tests/sec "
                          f"ETA: {eta/60:.1f} min")
                    print(f"  Success: {self.stats['successful_tests']}/{test_count} "
                          f"({self.stats['successful_tests']/test_count*100:.1f}%)")
                    print(f"  Perfect reversions: {self.stats['perfect_reversions']}/{test_count} "
                          f"({self.stats['perfect_reversions']/test_count*100:.1f}%)")
                    print()
        
        # Final statistics
        total_time = time.time() - start_time
        self.stats['total_time'] = total_time
        self.stats['tests_per_second'] = test_count / total_time
        
        return self._generate_comprehensive_report()
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        
        # Analyze error patterns
        error_analysis = {}
        for error_type, error_results in self.error_patterns.items():
            error_analysis[error_type] = {
                'count': len(error_results),
                'percentage': len(error_results) / self.stats['total_tests'] * 100,
                'common_augmentations': self._find_common_patterns(
                    [r.augmentations for r in error_results]
                )
            }
        
        # Find most problematic augmentation combinations
        failed_combinations = [r.augmentations for r in self.results if not r.success]
        problematic_combos = self._find_common_patterns(failed_combinations, top_k=20)
        
        # Find most successful combinations
        successful_combinations = [r.augmentations for r in self.results if r.perfect_reversion]
        successful_combos = self._find_common_patterns(successful_combinations, top_k=20)
        
        # Analyze execution times
        execution_times = [r.execution_time for r in self.results if r.execution_time > 0]
        time_stats = {
            'mean': sum(execution_times) / len(execution_times) if execution_times else 0,
            'min': min(execution_times) if execution_times else 0,
            'max': max(execution_times) if execution_times else 0
        }
        
        # Size analysis
        size_issues = []
        for result in self.results:
            if result.original_size and result.reverted_size:
                if result.original_size != result.reverted_size:
                    size_issues.append(result)
        
        report = {
            'summary': {
                'total_tests': self.stats['total_tests'],
                'successful_tests': self.stats['successful_tests'],
                'failed_tests': self.stats['failed_tests'],
                'perfect_reversions': self.stats['perfect_reversions'],
                'success_rate': self.stats['successful_tests'] / self.stats['total_tests'] * 100,
                'perfect_reversion_rate': self.stats['perfect_reversions'] / self.stats['total_tests'] * 100,
                'total_time': self.stats['total_time'],
                'tests_per_second': self.stats['tests_per_second']
            },
            'error_analysis': error_analysis,
            'problematic_combinations': problematic_combos,
            'successful_combinations': successful_combos,
            'execution_time_stats': time_stats,
            'size_issues': {
                'count': len(size_issues),
                'examples': [r.to_dict() for r in size_issues[:10]]  # First 10 examples
            },
            'detailed_results': [r.to_dict() for r in self.results]
        }
        
        return report
    
    def _find_common_patterns(self, combinations: List[List[str]], top_k: int = 10) -> List[Tuple[str, int]]:
        """Find the most common patterns in combinations."""
        # Count individual augmentations
        aug_counter = Counter()
        for combo in combinations:
            for aug in combo:
                aug_counter[aug] += 1
        
        # Count combination patterns
        combo_counter = Counter()
        for combo in combinations:
            combo_str = " + ".join(sorted(combo))
            combo_counter[combo_str] += 1
        
        return {
            'individual_augmentations': aug_counter.most_common(top_k),
            'combination_patterns': combo_counter.most_common(top_k)
        }
    
    def save_results(self, output_file: str):
        """Save test results to a JSON file."""
        report = self._generate_comprehensive_report()
        report['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': self.data_dir,
            'max_problems': self.max_problems,
            'total_problems_tested': len(self.all_problems),
            'eval_problems': len(self.eval_problems),
            'train_problems': len(self.train_problems)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of test results."""
        if not self.results:
            print("No test results available.")
            return
        
        print("="*80)
        print("AIRV COMPREHENSIVE TEST SUMMARY")
        print("="*80)
        
        total = self.stats['total_tests']
        successful = self.stats['successful_tests']
        perfect = self.stats['perfect_reversions']
        failed = self.stats['failed_tests']
        
        print(f"Total Tests: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Perfect Reversions: {perfect} ({perfect/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Execution Time: {self.stats['total_time']:.1f} seconds")
        print(f"Tests per Second: {self.stats['tests_per_second']:.1f}")
        print()
        
        # Error breakdown
        if self.error_patterns:
            print("ERROR BREAKDOWN:")
            for error_type, results in self.error_patterns.items():
                count = len(results)
                print(f"  {error_type}: {count} ({count/total*100:.1f}%)")
            print()
        
        # Most problematic combinations
        failed_combinations = [r.augmentations for r in self.results if not r.success]
        if failed_combinations:
            problematic = self._find_common_patterns(failed_combinations, top_k=5)
            print("MOST PROBLEMATIC AUGMENTATIONS:")
            for aug, count in problematic['individual_augmentations']:
                print(f"  {aug}: {count} failures")
            print()
        
        # Size issues
        size_issues = [r for r in self.results if r.original_size and r.reverted_size 
                      and r.original_size != r.reverted_size]
        if size_issues:
            print(f"SIZE MISMATCH ISSUES: {len(size_issues)}")
            for result in size_issues[:5]:  # Show first 5
                print(f"  {result.problem_id}: {result.original_size} -> {result.reverted_size}")
                print(f"    Augmentations: {result.augmentations}")
            print()
        
        print("="*80)


def main():
    """Main function to run comprehensive AIRV testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive AIRV Testing Framework")
    parser.add_argument("--data_dir", default=".", help="Directory containing ARC datasets")
    parser.add_argument("--max_problems", type=int, default=50, help="Maximum number of problems to test")
    parser.add_argument("--max_combinations", type=int, default=1000, help="Maximum augmentation combinations")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
    
    # Initialize tester
    tester = AIRVComprehensiveTester(
        data_dir=args.data_dir,
        max_problems=args.max_problems,
        verbose=not args.quiet
    )
    
    # Run tests
    try:
        results = tester.run_comprehensive_tests(max_combinations=args.max_combinations)
        
        # Print summary
        tester.print_summary()
        
        # Save results if requested
        if args.output:
            tester.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        tester.print_summary()
        if args.output:
            tester.save_results(args.output)
    
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
