#!/usr/bin/env python3
"""
AIRV Issue Analyzer - Analyzes remaining issues and provides specific fixes

This script analyzes the comprehensive test results to identify specific
patterns in failures and provide targeted fixes.
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


class AIRVIssueAnalyzer:
    """Analyzes AIRV test results to identify patterns and suggest fixes."""
    
    def __init__(self, results_file: str):
        """Initialize with results from comprehensive testing."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.detailed_results = self.results['detailed_results']
        self.failed_results = [r for r in self.detailed_results if not r['success']]
        self.partial_results = [r for r in self.detailed_results if r['success'] and not r['perfect_reversion']]
    
    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in failed tests."""
        print("="*80)
        print("ANALYZING FAILURE PATTERNS")
        print("="*80)
        
        # Error type analysis
        error_types = Counter()
        error_messages = defaultdict(list)
        
        for result in self.failed_results:
            error_type = result.get('error_type', 'Unknown')
            error_msg = result.get('error_message', 'Unknown error')
            error_types[error_type] += 1
            error_messages[error_type].append(error_msg)
        
        print(f"Total failures: {len(self.failed_results)}")
        print(f"Error types: {dict(error_types)}")
        
        # Analyze augmentation patterns in failures
        failed_augmentations = []
        for result in self.failed_results:
            failed_augmentations.extend(result['augmentations'])
        
        aug_failure_counts = Counter(failed_augmentations)
        print(f"\nMost problematic augmentations:")
        for aug, count in aug_failure_counts.most_common(10):
            print(f"  {aug}: {count} failures")
        
        # Analyze specific error messages
        print(f"\nDetailed error analysis:")
        for error_type, messages in error_messages.items():
            print(f"\n{error_type} ({len(messages)} occurrences):")
            # Show unique error messages
            unique_messages = list(set(messages))[:5]
            for msg in unique_messages:
                print(f"  - {msg}")
        
        return {
            'total_failures': len(self.failed_results),
            'error_types': dict(error_types),
            'problematic_augmentations': dict(aug_failure_counts.most_common(10)),
            'error_messages': {k: list(set(v))[:5] for k, v in error_messages.items()}
        }
    
    def analyze_partial_successes(self) -> Dict[str, Any]:
        """Analyze patterns in tests that succeeded but didn't have perfect reversion."""
        print("\n" + "="*80)
        print("ANALYZING PARTIAL SUCCESSES (Non-perfect reversions)")
        print("="*80)
        
        print(f"Total partial successes: {len(self.partial_results)}")
        
        # Size mismatch analysis
        size_mismatches = []
        for result in self.partial_results:
            orig_size = result.get('original_size')
            rev_size = result.get('reverted_size')
            if orig_size and rev_size and orig_size != rev_size:
                size_mismatches.append({
                    'problem_id': result['problem_id'],
                    'augmentations': result['augmentations'],
                    'original_size': orig_size,
                    'reverted_size': rev_size
                })
        
        print(f"Size mismatches: {len(size_mismatches)}")
        if size_mismatches:
            print("Examples:")
            for example in size_mismatches[:5]:
                print(f"  Problem {example['problem_id']}: {example['original_size']} -> {example['reverted_size']}")
                print(f"    Augmentations: {example['augmentations']}")
        
        # Augmentation patterns in partial successes
        partial_augmentations = []
        for result in self.partial_results:
            partial_augmentations.extend(result['augmentations'])
        
        aug_partial_counts = Counter(partial_augmentations)
        print(f"\nAugmentations in partial successes:")
        for aug, count in aug_partial_counts.most_common(10):
            print(f"  {aug}: {count} occurrences")
        
        return {
            'total_partial': len(self.partial_results),
            'size_mismatches': len(size_mismatches),
            'size_mismatch_examples': size_mismatches[:10],
            'partial_augmentations': dict(aug_partial_counts.most_common(10))
        }
    
    def identify_upscale_issues(self) -> Dict[str, Any]:
        """Specifically analyze upscale-related issues."""
        print("\n" + "="*80)
        print("ANALYZING UPSCALE ISSUES")
        print("="*80)
        
        upscale_failures = []
        upscale_partials = []
        
        # Find all upscale-related failures
        for result in self.failed_results:
            if any('upscale' in aug for aug in result['augmentations']):
                upscale_failures.append(result)
        
        # Find all upscale-related partial successes
        for result in self.partial_results:
            if any('upscale' in aug for aug in result['augmentations']):
                upscale_partials.append(result)
        
        print(f"Upscale failures: {len(upscale_failures)}")
        print(f"Upscale partial successes: {len(upscale_partials)}")
        
        # Analyze upscale failure patterns
        if upscale_failures:
            print("\nUpscale failure analysis:")
            for result in upscale_failures[:5]:
                print(f"  Problem {result['problem_id']}: {result['error_message']}")
                print(f"    Augmentations: {result['augmentations']}")
        
        # Analyze upscale size issues
        upscale_size_issues = []
        for result in upscale_partials:
            orig_size = result.get('original_size')
            aug_size = result.get('augmented_size')
            rev_size = result.get('reverted_size')
            
            if orig_size and rev_size and orig_size != rev_size:
                upscale_size_issues.append({
                    'problem_id': result['problem_id'],
                    'original_size': orig_size,
                    'augmented_size': aug_size,
                    'reverted_size': rev_size,
                    'augmentations': result['augmentations']
                })
        
        print(f"\nUpscale size issues: {len(upscale_size_issues)}")
        if upscale_size_issues:
            print("Examples:")
            for issue in upscale_size_issues[:5]:
                print(f"  {issue['original_size']} -> {issue['augmented_size']} -> {issue['reverted_size']}")
                print(f"    Augmentations: {issue['augmentations']}")
        
        return {
            'upscale_failures': len(upscale_failures),
            'upscale_partials': len(upscale_partials),
            'upscale_size_issues': len(upscale_size_issues),
            'size_issue_examples': upscale_size_issues[:10]
        }
    
    def suggest_specific_fixes(self, failure_analysis: Dict[str, Any],
                             partial_analysis: Dict[str, Any],
                             upscale_analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific fixes based on analysis."""
        fixes = []
        
        # Upscale-specific fixes
        if upscale_analysis['upscale_failures'] > 0:
            fixes.append(
                f"CRITICAL FIX: {upscale_analysis['upscale_failures']} upscale operations are failing completely. "
                "This suggests issues with the upscale augmentation itself, not just deaugmentation. "
                "Check for edge cases where target size is invalid or grid processing fails."
            )
        
        if upscale_analysis['upscale_size_issues'] > 0:
            fixes.append(
                f"UPSCALE FIX: {upscale_analysis['upscale_size_issues']} upscale operations have size reversion issues. "
                "The position detection heuristic in deupscale_grid() needs improvement. "
                "Consider storing the actual placement position during upscaling."
            )
        
        # Size mismatch fixes
        if partial_analysis['size_mismatches'] > 0:
            fixes.append(
                f"SIZE FIX: {partial_analysis['size_mismatches']} tests have size mismatches after reversion. "
                "This indicates issues with geometric transformations combined with upscaling. "
                "Ensure that size tracking is maintained through transformation sequences."
            )
        
        # Error-specific fixes
        error_types = failure_analysis['error_types']
        if 'AUGMENT_ERROR' in error_types:
            fixes.append(
                f"AUGMENT FIX: {error_types['AUGMENT_ERROR']} augmentation operations are failing. "
                "Add better error handling and validation in the augmentation pipeline. "
                "Check for invalid grid sizes, empty grids, or malformed data."
            )
        
        # General improvements
        fixes.append(
            "ROBUSTNESS FIX: Add comprehensive validation at each step of the augmentation pipeline. "
            "Validate grid sizes, check for empty grids, and ensure all metadata is properly structured."
        )
        
        fixes.append(
            "METADATA FIX: Standardize the metadata structure across all augmentation types. "
            "Ensure consistent parameter passing between augmentation and deaugmentation functions."
        )
        
        fixes.append(
            "POSITION FIX: For upscale operations, store the actual placement position during augmentation "
            "instead of relying on heuristic detection during deaugmentation. This will eliminate most "
            "upscale-related reversion issues."
        )
        
        return fixes
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE AIRV ANALYSIS REPORT")
        print("="*80)
        
        # Run all analyses
        failure_analysis = self.analyze_failure_patterns()
        partial_analysis = self.analyze_partial_successes()
        upscale_analysis = self.identify_upscale_issues()
        
        # Generate fixes
        suggested_fixes = self.suggest_specific_fixes(
            failure_analysis, partial_analysis, upscale_analysis
        )
        
        # Print suggested fixes
        print("\n" + "="*80)
        print("SUGGESTED FIXES")
        print("="*80)
        for i, fix in enumerate(suggested_fixes, 1):
            print(f"\n{i}. {fix}")
        
        # Calculate improvement metrics
        total_tests = self.results['summary']['total_tests']
        current_success_rate = self.results['summary']['success_rate']
        current_perfect_rate = self.results['summary']['perfect_reversion_rate']
        
        # Estimate potential improvement
        fixable_failures = failure_analysis.get('error_types', {}).get('AUGMENT_ERROR', 0)
        fixable_partials = upscale_analysis['upscale_size_issues']
        
        potential_success_rate = (self.results['summary']['successful_tests'] + fixable_failures) / total_tests * 100
        potential_perfect_rate = (self.results['summary']['perfect_reversions'] + fixable_partials) / total_tests * 100
        
        print(f"\n" + "="*80)
        print("IMPROVEMENT POTENTIAL")
        print("="*80)
        print(f"Current success rate: {current_success_rate:.1f}%")
        print(f"Potential success rate: {potential_success_rate:.1f}% (+{potential_success_rate - current_success_rate:.1f}%)")
        print(f"Current perfect reversion rate: {current_perfect_rate:.1f}%")
        print(f"Potential perfect reversion rate: {potential_perfect_rate:.1f}% (+{potential_perfect_rate - current_perfect_rate:.1f}%)")
        
        return {
            'failure_analysis': failure_analysis,
            'partial_analysis': partial_analysis,
            'upscale_analysis': upscale_analysis,
            'suggested_fixes': suggested_fixes,
            'improvement_potential': {
                'current_success_rate': current_success_rate,
                'potential_success_rate': potential_success_rate,
                'current_perfect_rate': current_perfect_rate,
                'potential_perfect_rate': potential_perfect_rate
            }
        }


def main():
    """Main function for AIRV issue analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIRV Issue Analyzer")
    parser.add_argument("results_file", help="JSON file with comprehensive test results")
    parser.add_argument("--output", help="Output file for analysis report (JSON)")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = AIRVIssueAnalyzer(args.results_file)
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nAnalysis report saved to {args.output}")
        
    except FileNotFoundError:
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file '{args.results_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
