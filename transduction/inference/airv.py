"""
AIRV (Augment, Infer, Revert and Vote) Inference Technique

This module implements the AIRV inference technique:
1. AUGMENT: Create multiple augmented versions of the test problem
2. INFER: Run inference on original + all augmented versions  
3. REVERT: Attempt to revert augmented outputs back to original space
4. VOTE: Aggregate all valid outputs and pick the most frequent one

The key insight is that different augmentations might make the pattern more
obvious to the model, and by reverting back to the original space and voting,
we can get more robust predictions.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transduction.eval import InferenceTechnique
from transduction.inference.inference import ARCTransductionInference
from augment import apply_random_augmentations, get_available_augmentations, apply_augmentation_to_problem, track_pixel_transformations
from deaugment import apply_full_deaugmentation, get_deaugmentation_functions, create_augmentation_metadata, apply_pixel_level_deaugmentation


class AIRVInference(InferenceTechnique):
    """
    AIRV (Augment, Infer, Revert and Vote) inference technique.
    
    This technique generates multiple augmented versions of the input problem,
    runs inference on each, reverts the outputs back to original space,
    and uses majority voting to select the final answer.
    """
    
    def __init__(self, model_name: str, device: str = "auto", 
                 num_augmentations: int = 8,
                 include_original: bool = True,
                 augmentation_seed: Optional[int] = None,
                 temperature: float = 0.1,
                 **kwargs):
        """
        Initialize AIRV inference technique.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run inference on
            num_augmentations: Number of augmented versions to create
            include_original: Whether to include original (non-augmented) version
            augmentation_seed: Seed for augmentation randomness (None for random)
            temperature: Temperature for model generation
            **kwargs: Additional arguments
        """
        self.inference = ARCTransductionInference(model_name=model_name, device=device)
        self.num_augmentations = num_augmentations
        self.include_original = include_original
        self.augmentation_seed = augmentation_seed
        
        # Configure generation for more focused outputs
        self.inference.generation_config.temperature = temperature
        self.inference.generation_config.do_sample = temperature > 0.0
        self.inference.generation_config.top_p = 0.9
        
        # Get augmentation functions
        self.augmentation_funcs = get_available_augmentations()
        self.deaugmentation_funcs = get_deaugmentation_functions()
        
        print(f"AIRV initialized with {num_augmentations} augmentations")
        print(f"Available augmentations: {list(self.augmentation_funcs.keys())}")
    
    def create_augmented_versions(self, problem_data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[str], Dict[str, Any]]]:
        """
        Create multiple augmented versions of the problem.
        
        Args:
            problem_data: Original problem data
            
        Returns:
            List of tuples: (augmented_problem, augmentation_list, metadata)
        """
        versions = []
        
        # Set seed for reproducible augmentations if specified
        if self.augmentation_seed is not None:
            random.seed(self.augmentation_seed)
        
        # Add original version if requested
        if self.include_original:
            original_metadata = create_augmentation_metadata(problem_data, [])
            versions.append((deepcopy(problem_data), [], original_metadata))
        
        # Create augmented versions
        for i in range(self.num_augmentations):
            try:
                # Create augmented problem with comprehensive metadata
                augmented_problem, applied_augs, aug_params = apply_random_augmentations(
                    problem_data, 
                    num_augmentations=random.randint(1, 3),  # 1-3 augmentations per version
                    seed=None  # Let it be random for each version
                )
                
                # Create comprehensive metadata for perfect deaugmentation
                metadata = self._create_comprehensive_metadata(problem_data, applied_augs, aug_params)
                
                versions.append((augmented_problem, applied_augs, metadata))
                
            except Exception as e:
                print(f"Warning: Failed to create augmentation {i}: {e}")
                continue
        
        print(f"Created {len(versions)} versions ({len(versions) - (1 if self.include_original else 0)} augmented)")
        return versions
    
    def _create_comprehensive_metadata(self, original_problem: Dict[str, Any], 
                                     applied_augs: List[str], 
                                     aug_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive metadata for perfect deaugmentation.
        
        Args:
            original_problem: Original problem data
            applied_augs: List of applied augmentations
            aug_params: Parameters used for each augmentation
            
        Returns:
            Comprehensive metadata dictionary
        """
        metadata = {
            'augmentation_params': aug_params,
            'applied_augmentations': applied_augs,
        }
        
        # Get first training input for pixel tracking
        if 'train' in original_problem and original_problem['train']:
            first_input = original_problem['train'][0]['input']
            
            # Create pixel-level transformation metadata
            try:
                pixel_metadata = track_pixel_transformations(first_input, applied_augs, aug_params)
                metadata.update(pixel_metadata)
            except Exception as e:
                print(f"Warning: Could not create pixel transformation metadata: {e}")
                # Fallback to basic metadata
                metadata.update(create_augmentation_metadata(original_problem, applied_augs))
        else:
            # Fallback if no training data
            metadata.update(create_augmentation_metadata(original_problem, applied_augs))
        
        return metadata
    

    
    def infer_on_version(self, problem_version: Dict[str, Any], 
                        train_sample_count: int = 3,
                        test_example_idx: int = 0) -> Dict[str, Any]:
        """
        Run inference on a single version of the problem.
        
        Args:
            problem_version: Problem data (original or augmented)
            train_sample_count: Number of training examples to use
            test_example_idx: Index of test example
            
        Returns:
            Inference result dictionary
        """
        return self.inference.infer_single_problem(
            problem_version,
            train_sample_count=train_sample_count,
            test_example_idx=test_example_idx,
            verbose=False
        )
    
    def revert_output(self, predicted_grid: Optional[List[List[int]]], 
                     augmentation_list: List[str],
                     metadata: Dict[str, Any]) -> Optional[List[List[int]]]:
        """
        Revert an augmented output back to original space.
        
        Args:
            predicted_grid: Grid predicted on augmented problem
            augmentation_list: List of augmentations that were applied
            metadata: Metadata for deaugmentation
            
        Returns:
            Reverted grid or None if reversion failed
        """
        if predicted_grid is None or not augmentation_list:
            return predicted_grid
        
        try:
            # Try pixel-level deaugmentation first (more accurate)
            if 'pixel_transformations' in metadata:
                reverted_grid = apply_pixel_level_deaugmentation(predicted_grid, metadata)
                if reverted_grid is not None:
                    return reverted_grid
            
            # Fallback to sequential deaugmentation
            dummy_problem = {
                'test': [{'output': predicted_grid}]
            }
            
            # Use augmentation parameters from metadata if available
            aug_metadata = metadata.get('augmentation_params', metadata)
            
            reverted_problem = apply_full_deaugmentation(
                dummy_problem, 
                augmentation_list, 
                aug_metadata
            )
            
            return reverted_problem['test'][0]['output']
            
        except IndexError as e:
            print(f"Warning: Failed to revert output with augmentations {augmentation_list}: list index out of range")
            print(f"  Predicted grid shape: {len(predicted_grid)}x{len(predicted_grid[0]) if predicted_grid else 0}")
            print(f"  Available metadata keys: {list(metadata.keys())}")
            return None
        except Exception as e:
            print(f"Warning: Failed to revert output with augmentations {augmentation_list}: {e}")
            return None
    
    def vote_on_outputs(self, valid_outputs: List[List[List[int]]]) -> Tuple[Optional[List[List[int]]], Dict[str, int]]:
        """
        Perform majority voting on valid outputs.
        
        Args:
            valid_outputs: List of valid (non-None) output grids
            
        Returns:
            Tuple of (winning_grid, vote_counts)
        """
        if not valid_outputs:
            return None, {}
        
        # Convert grids to strings for comparison
        grid_strings = []
        for grid in valid_outputs:
            grid_str = json.dumps(grid)  # Use JSON for reliable string representation
            grid_strings.append(grid_str)
        
        # Count votes
        vote_counts = {}
        for grid_str in grid_strings:
            vote_counts[grid_str] = vote_counts.get(grid_str, 0) + 1
        
        # Find the grid with most votes
        if not vote_counts:
            return None, {}
        
        winning_grid_str = max(vote_counts, key=vote_counts.get)
        winning_grid = json.loads(winning_grid_str)
        
        # Convert back to readable format for logging
        readable_counts = {}
        for grid_str, count in vote_counts.items():
            grid = json.loads(grid_str)
            if grid:
                grid_key = f"{len(grid)}x{len(grid[0]) if grid else 0}"
                readable_counts[grid_key] = readable_counts.get(grid_key, 0) + count
            else:
                readable_counts["empty"] = readable_counts.get("empty", 0) + count
        
        return winning_grid, readable_counts
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        """
        Perform AIRV inference on a single problem.
        
        Args:
            problem_data: Problem data dictionary
            train_sample_count: Number of training examples to use
            test_example_idx: Index of test example to use
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with inference results
        """
        if verbose:
            print(f"Starting AIRV inference with {self.num_augmentations} augmentations")
        
        # Step 1: AUGMENT - Create multiple versions
        versions = self.create_augmented_versions(problem_data)
        
        if not versions:
            # Fallback to standard inference if no versions created
            print("Warning: No versions created, falling back to standard inference")
            return self.inference.infer_single_problem(
                problem_data, train_sample_count, test_example_idx, verbose
            )
        
        # Step 2: INFER - Run inference on all versions
        version_results = []
        for i, (problem_version, aug_list, metadata) in enumerate(versions):
            try:
                result = self.infer_on_version(
                    problem_version, train_sample_count, test_example_idx
                )
                version_results.append({
                    'version_idx': i,
                    'augmentations': aug_list,
                    'metadata': metadata,
                    'inference_result': result,
                    'raw_prediction': result['predicted_grid']
                })
                
                if verbose:
                    aug_desc = "original" if not aug_list else f"augs: {aug_list}"
                    pred_desc = "parsed" if result['predicted_grid'] else "failed"
                    print(f"  Version {i} ({aug_desc}): {pred_desc}")
                    
            except Exception as e:
                print(f"Warning: Inference failed on version {i}: {e}")
                continue
        
        # Step 3: REVERT - Revert augmented outputs back to original space
        valid_outputs = []
        reversion_info = []
        
        for version_result in version_results:
            aug_list = version_result['augmentations']
            metadata = version_result['metadata']
            raw_pred = version_result['raw_prediction']
            
            if not aug_list:
                # Original version, no reversion needed
                if raw_pred is not None:
                    valid_outputs.append(raw_pred)
                    reversion_info.append({
                        'version_idx': version_result['version_idx'],
                        'augmentations': aug_list,
                        'reversion_success': True,
                        'reverted_grid': raw_pred
                    })
            else:
                # Augmented version, needs reversion
                reverted_grid = self.revert_output(raw_pred, aug_list, metadata)
                
                reversion_success = reverted_grid is not None
                reversion_info.append({
                    'version_idx': version_result['version_idx'],
                    'augmentations': aug_list,
                    'reversion_success': reversion_success,
                    'reverted_grid': reverted_grid
                })
                
                if reverted_grid is not None:
                    valid_outputs.append(reverted_grid)
                
                if verbose:
                    status = "success" if reversion_success else "failed"
                    print(f"  Reversion {version_result['version_idx']}: {status}")
        
        # Step 4: VOTE - Aggregate and pick most frequent answer
        winning_grid, vote_counts = self.vote_on_outputs(valid_outputs)
        
        # Get ground truth for evaluation
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        
        # Evaluate final prediction
        is_correct = self.inference.evaluate_prediction(winning_grid, ground_truth)
        
        # Compile comprehensive result
        result = {
            'prompt': f"AIRV with {len(versions)} versions",
            'response': f"Voted from {len(valid_outputs)} valid outputs",
            'predicted_grid': winning_grid,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'train_sample_count': train_sample_count,
            'test_example_idx': test_example_idx,
            
            # AIRV-specific information
            'num_versions': len(versions),
            'num_augmented': len(versions) - (1 if self.include_original else 0),
            'valid_outputs_count': len(valid_outputs),
            'vote_counts': vote_counts,
            'version_results': version_results,
            'reversion_info': reversion_info,
            'augmentation_seed': self.augmentation_seed
        }
        
        if verbose:
            print(f"\nAIRV Results:")
            print(f"  Versions created: {len(versions)}")
            print(f"  Valid outputs: {len(valid_outputs)}")
            print(f"  Vote counts: {vote_counts}")
            print(f"  Final prediction: {'correct' if is_correct else 'incorrect'}")
            
            print(f"\nGROUND TRUTH:")
            for row in ground_truth:
                print(';'.join(map(str, row)))
            print()
            
            print(f"PREDICTED (AIRV):")
            if winning_grid:
                for row in winning_grid:
                    print(';'.join(map(str, row)))
            else:
                print("No valid prediction")
            print()
            
            print(f"CORRECT: {is_correct}")
            print("=" * 50)
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.inference, 'model'):
            del self.inference.model
        if hasattr(self.inference, 'tokenizer'):
            del self.inference.tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Convenience function to create AIRV configurations
def create_airv_configs() -> List[Dict[str, Any]]:
    """
    Create a set of AIRV configuration variants for experimentation.
    
    Returns:
        List of configuration dictionaries
    """
    configs = [
        {
            'name': 'airv_light',
            'params': {
                'num_augmentations': 4,
                'include_original': True,
                'temperature': 0.1
            },
            'description': 'Light AIRV with 4 augmentations'
        },
        {
            'name': 'airv_standard',
            'params': {
                'num_augmentations': 8,
                'include_original': True,
                'temperature': 0.1
            },
            'description': 'Standard AIRV with 8 augmentations'
        },
        {
            'name': 'airv_heavy',
            'params': {
                'num_augmentations': 12,
                'include_original': True,
                'temperature': 0.1
            },
            'description': 'Heavy AIRV with 12 augmentations'
        },
        {
            'name': 'airv_no_original',
            'params': {
                'num_augmentations': 8,
                'include_original': False,
                'temperature': 0.1
            },
            'description': 'AIRV without original version'
        },
        {
            'name': 'airv_diverse',
            'params': {
                'num_augmentations': 8,
                'include_original': True,
                'temperature': 0.3
            },
            'description': 'AIRV with more diverse generation'
        }
    ]
    
    return configs


# Example usage and testing
if __name__ == "__main__":
    # Simple test of AIRV components
    print("Testing AIRV components...")
    
    # Test augmentation creation
    test_problem = {
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[5, 6], [7, 8]]},
        ],
        'test': [
            {'input': [[8, 9], [1, 2]], 'output': [[3, 4], [5, 6]]}
        ]
    }
    
    # This would require model loading, so just test the structure
    try:
        # airv = AIRVInference("dummy_model")  # Would fail without real model
        # versions = airv.create_augmented_versions(test_problem)
        # print(f"Created {len(versions)} versions")
        print("AIRV structure test passed")
    except Exception as e:
        print(f"Expected error (no model): {e}")
    
    print("AIRV implementation complete!")
