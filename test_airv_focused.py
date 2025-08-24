#!/usr/bin/env python3
"""
Focused AIRV Testing - Identifies and fixes specific issues

This script focuses on identifying and fixing specific AIRV issues by:
1. Testing each augmentation individually 
2. Testing problematic combinations
3. Providing detailed debugging information
4. Suggesting fixes for common issues
"""

import json
import random
import traceback
from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augment import (
    apply_random_augmentations, 
    get_available_augmentations, 
    apply_augmentation_to_problem,
    rotate_90, rotate_180, rotate_270,
    flip_vertical, flip_horizontal,
    apply_color_permutation, upscale_grid
)
from deaugment import (
    apply_full_deaugmentation, 
    apply_pixel_level_deaugmentation,
    create_augmentation_metadata,
    get_deaugmentation_functions,
    derotate_90, derotate_180, derotate_270,
    deflip_vertical, deflip_horizontal,
    deapply_color_permutation, deupscale_grid
)
from loader import load_evaluation_problem, load_training_problem


class AIRVFocusedTester:
    """Focused tester for identifying specific AIRV issues."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.augmentation_funcs = get_available_augmentations()
        self.deaugmentation_funcs = get_deaugmentation_functions()
        
        # Create a simple test problem
        self.test_problem = {
            'train': [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
                }
            ],
            'test': [
                {
                    'input': [[0, 1, 2], [3, 4, 5]]
                }
            ]
        }
    
    def test_individual_augmentations(self) -> Dict[str, Any]:
        """Test each augmentation individually to identify issues."""
        results = {}
        
        if self.verbose:
            print("="*60)
            print("TESTING INDIVIDUAL AUGMENTATIONS")
            print("="*60)
        
        for aug_name, aug_func in self.augmentation_funcs.items():
            if self.verbose:
                print(f"\nTesting: {aug_name}")
                print("-" * 40)
            
            result = self._test_single_augmentation(aug_name)
            results[aug_name] = result
            
            if self.verbose:
                if result['success']:
                    status = "✓ SUCCESS" if result['perfect_reversion'] else "⚠ PARTIAL"
                    print(f"Status: {status}")
                    if not result['perfect_reversion']:
                        print(f"Issue: {result['issue']}")
                else:
                    print(f"Status: ✗ FAILED")
                    print(f"Error: {result['error']}")
        
        return results
    
    def _test_single_augmentation(self, aug_name: str) -> Dict[str, Any]:
        """Test a single augmentation and its reversion."""
        result = {
            'augmentation': aug_name,
            'success': False,
            'perfect_reversion': False,
            'error': None,
            'issue': None,
            'original_grid': None,
            'augmented_grid': None,
            'reverted_grid': None,
            'metadata': {}
        }
        
        try:
            original_problem = deepcopy(self.test_problem)
            original_grid = original_problem['train'][0]['input']
            result['original_grid'] = original_grid
            
            # Apply augmentation
            if aug_name == 'color_permutation':
                # Create a specific color map for testing
                color_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 0}
                augmented_problem = apply_augmentation_to_problem(
                    original_problem, self.augmentation_funcs[aug_name], color_map=color_map
                )
                aug_params = {aug_name: {'color_map': color_map}}
                
            elif aug_name == 'upscale':
                target_size = (5, 5)  # Fixed size for testing
                augmented_problem = apply_augmentation_to_problem(
                    original_problem, self.augmentation_funcs[aug_name], target_size=target_size
                )
                upscale_name = f"upscale_{target_size[0]}x{target_size[1]}"
                aug_params = {upscale_name: {
                    'original_size': (len(original_grid), len(original_grid[0])),
                    'target_size': target_size
                }}
                aug_name = upscale_name  # Update name for deaugmentation
                
            else:
                # Simple augmentation
                augmented_problem = apply_augmentation_to_problem(
                    original_problem, self.augmentation_funcs[aug_name]
                )
                aug_params = {}
            
            augmented_grid = augmented_problem['train'][0]['input']
            result['augmented_grid'] = augmented_grid
            
            # Create metadata
            metadata = create_augmentation_metadata(original_problem, [aug_name])
            metadata['augmentation_params'] = aug_params
            result['metadata'] = metadata
            
            # Apply deaugmentation
            reverted_problem = apply_full_deaugmentation(
                augmented_problem, [aug_name], metadata
            )
            
            reverted_grid = reverted_problem['train'][0]['input']
            result['reverted_grid'] = reverted_grid
            
            # Check if reversion is perfect
            result['perfect_reversion'] = (original_grid == reverted_grid)
            result['success'] = True
            
            if not result['perfect_reversion']:
                result['issue'] = self._diagnose_reversion_issue(
                    original_grid, augmented_grid, reverted_grid, aug_name
                )
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _diagnose_reversion_issue(self, original: List[List[int]], 
                                augmented: List[List[int]], 
                                reverted: List[List[int]], 
                                aug_name: str) -> str:
        """Diagnose what went wrong with reversion."""
        
        orig_size = (len(original), len(original[0]))
        aug_size = (len(augmented), len(augmented[0]))
        rev_size = (len(reverted), len(reverted[0]))
        
        issues = []
        
        # Size issues
        if orig_size != rev_size:
            issues.append(f"Size mismatch: {orig_size} -> {rev_size}")
        
        # Content issues
        if orig_size == rev_size:
            diff_positions = []
            for i in range(len(original)):
                for j in range(len(original[0])):
                    if original[i][j] != reverted[i][j]:
                        diff_positions.append((i, j))
            
            if diff_positions:
                issues.append(f"Content differs at {len(diff_positions)} positions")
                if len(diff_positions) <= 5:
                    issues.append(f"Diff positions: {diff_positions}")
        
        # Specific augmentation issues
        if aug_name.startswith('upscale'):
            issues.append("Upscale reversion issue - possibly incorrect position detection")
        elif aug_name == 'color_permutation':
            issues.append("Color permutation issue - possibly missing or incorrect color map")
        
        return "; ".join(issues) if issues else "Unknown issue"
    
    def test_problematic_combinations(self) -> Dict[str, Any]:
        """Test known problematic combinations."""
        if self.verbose:
            print("="*60)
            print("TESTING PROBLEMATIC COMBINATIONS")
            print("="*60)
        
        problematic_combos = [
            ['color_permutation', 'rotate_90'],
            ['upscale', 'flip_horizontal'],
            ['rotate_180', 'color_permutation'],
            ['flip_vertical', 'upscale'],
            ['rotate_270', 'color_permutation', 'flip_horizontal'],
            ['upscale', 'rotate_90', 'color_permutation']
        ]
        
        results = {}
        
        for combo in problematic_combos:
            combo_name = " + ".join(combo)
            if self.verbose:
                print(f"\nTesting: {combo_name}")
                print("-" * 40)
            
            result = self._test_combination(combo)
            results[combo_name] = result
            
            if self.verbose:
                if result['success']:
                    status = "✓ SUCCESS" if result['perfect_reversion'] else "⚠ PARTIAL"
                    print(f"Status: {status}")
                    if not result['perfect_reversion']:
                        print(f"Issue: {result['issue']}")
                else:
                    print(f"Status: ✗ FAILED")
                    print(f"Error: {result['error']}")
        
        return results
    
    def _test_combination(self, combo: List[str]) -> Dict[str, Any]:
        """Test a combination of augmentations."""
        result = {
            'combination': combo,
            'success': False,
            'perfect_reversion': False,
            'error': None,
            'issue': None,
            'steps': []
        }
        
        try:
            current_problem = deepcopy(self.test_problem)
            applied_augs = []
            aug_params = {}
            
            # Apply each augmentation in sequence
            for aug_name in combo:
                step_result = {'augmentation': aug_name, 'success': False}
                
                if aug_name == 'color_permutation':
                    color_map = {i: (i + 1) % 10 for i in range(10)}
                    current_problem = apply_augmentation_to_problem(
                        current_problem, self.augmentation_funcs[aug_name], color_map=color_map
                    )
                    aug_params[aug_name] = {'color_map': color_map}
                    applied_augs.append(aug_name)
                    
                elif aug_name == 'upscale':
                    target_size = (6, 6)
                    current_problem = apply_augmentation_to_problem(
                        current_problem, self.augmentation_funcs[aug_name], target_size=target_size
                    )
                    upscale_name = f"upscale_{target_size[0]}x{target_size[1]}"
                    aug_params[upscale_name] = {
                        'original_size': (3, 3),  # Original test grid size
                        'target_size': target_size
                    }
                    applied_augs.append(upscale_name)
                    
                else:
                    current_problem = apply_augmentation_to_problem(
                        current_problem, self.augmentation_funcs[aug_name]
                    )
                    applied_augs.append(aug_name)
                
                step_result['success'] = True
                step_result['grid_size'] = (
                    len(current_problem['train'][0]['input']),
                    len(current_problem['train'][0]['input'][0])
                )
                result['steps'].append(step_result)
            
            # Create metadata and apply deaugmentation
            metadata = create_augmentation_metadata(self.test_problem, applied_augs)
            metadata['augmentation_params'] = aug_params
            
            reverted_problem = apply_full_deaugmentation(
                current_problem, applied_augs, metadata
            )
            
            # Check reversion
            original_grid = self.test_problem['train'][0]['input']
            reverted_grid = reverted_problem['train'][0]['input']
            
            result['perfect_reversion'] = (original_grid == reverted_grid)
            result['success'] = True
            
            if not result['perfect_reversion']:
                result['issue'] = self._diagnose_reversion_issue(
                    original_grid, current_problem['train'][0]['input'], 
                    reverted_grid, " + ".join(combo)
                )
            
        except Exception as e:
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def suggest_fixes(self, individual_results: Dict[str, Any], 
                     combination_results: Dict[str, Any]) -> List[str]:
        """Suggest fixes based on test results."""
        fixes = []
        
        # Analyze individual augmentation issues
        for aug_name, result in individual_results.items():
            if not result['success'] or not result['perfect_reversion']:
                if aug_name == 'color_permutation':
                    fixes.append(
                        "FIX: Color permutation metadata is not being passed correctly to deaugmentation. "
                        "Ensure augmentation_params contains the color_map."
                    )
                elif aug_name.startswith('upscale'):
                    fixes.append(
                        "FIX: Upscale deaugmentation needs better position detection. "
                        "Consider storing the actual placement position during augmentation."
                    )
        
        # Analyze combination issues
        color_perm_issues = 0
        upscale_issues = 0
        
        for combo_name, result in combination_results.items():
            if not result['perfect_reversion']:
                if 'color_permutation' in combo_name:
                    color_perm_issues += 1
                if 'upscale' in combo_name:
                    upscale_issues += 1
        
        if color_perm_issues > 0:
            fixes.append(
                f"FIX: Color permutation fails in {color_perm_issues} combinations. "
                "The metadata structure needs to be standardized for proper parameter passing."
            )
        
        if upscale_issues > 0:
            fixes.append(
                f"FIX: Upscale fails in {upscale_issues} combinations. "
                "Position detection heuristics need improvement or position should be stored."
            )
        
        # General fixes
        fixes.append(
            "IMPROVEMENT: Add comprehensive metadata validation to ensure all required "
            "parameters are available for deaugmentation."
        )
        
        fixes.append(
            "IMPROVEMENT: Implement more robust error handling that can continue "
            "deaugmentation even if one step fails."
        )
        
        return fixes
    
    def run_focused_tests(self) -> Dict[str, Any]:
        """Run all focused tests and return comprehensive results."""
        
        # Test individual augmentations
        individual_results = self.test_individual_augmentations()
        
        # Test problematic combinations
        combination_results = self.test_problematic_combinations()
        
        # Generate fixes
        suggested_fixes = self.suggest_fixes(individual_results, combination_results)
        
        # Print summary
        if self.verbose:
            print("="*60)
            print("FOCUSED TEST SUMMARY")
            print("="*60)
            
            # Individual results summary
            individual_success = sum(1 for r in individual_results.values() if r['perfect_reversion'])
            print(f"Individual Augmentations: {individual_success}/{len(individual_results)} perfect")
            
            # Combination results summary
            combo_success = sum(1 for r in combination_results.values() if r['perfect_reversion'])
            print(f"Problematic Combinations: {combo_success}/{len(combination_results)} perfect")
            
            print("\nSUGGESTED FIXES:")
            for i, fix in enumerate(suggested_fixes, 1):
                print(f"{i}. {fix}")
        
        return {
            'individual_results': individual_results,
            'combination_results': combination_results,
            'suggested_fixes': suggested_fixes,
            'summary': {
                'individual_success_rate': sum(1 for r in individual_results.values() if r['perfect_reversion']) / len(individual_results),
                'combination_success_rate': sum(1 for r in combination_results.values() if r['perfect_reversion']) / len(combination_results)
            }
        }


def main():
    """Main function for focused AIRV testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Focused AIRV Testing")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = AIRVFocusedTester(verbose=not args.quiet)
    
    # Run tests
    results = tester.run_focused_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
