"""Reward function for ARC Transduction task."""

from typing import Any, Dict, List

from transduction.metrics import check_array, check_value, parse_grid_from_string


def reward_function(
    completions: List[str], 
    expected_outputs: List[List[List[int]]], 
    **kwargs: Any
) -> List[float]:
    """
    Reward function for transduction task.
    
    Reward scheme:
    - -1.0: Invalid array format (cannot be parsed)
    - 0.0: Valid array format but incorrect result
    - 1.0: Valid array format and correct result
    
    Args:
        completions: List of model completions (raw output strings)
        expected_outputs: List of expected ground truth grids
        **kwargs: Additional arguments (unused)
        
    Returns:
        List of reward scores corresponding to each completion
        
    Examples:
        >>> reward_function(["12;34"], [[[1, 2], [3, 4]]])
        [1.0]
        >>> reward_function(["12;35"], [[[1, 2], [3, 4]]])
        [0.0]
        >>> reward_function(["invalid"], [[[1, 2], [3, 4]]])
        [-1.0]
    """
    rewards = []
    
    for completion, expected in zip(completions, expected_outputs, strict=False):
        # Check if the completion can be parsed into a valid array
        if not check_array(completion):
            # Invalid array format
            rewards.append(-1.0)
            continue
        
        # Valid array format, now check if the value is correct
        if check_value(completion, expected):
            # Correct result
            rewards.append(1.0)
        else:
            # Valid format but wrong result
            rewards.append(0.0)
    
    return rewards


def detailed_reward_function(
    completions: List[str], 
    expected_outputs: List[List[List[int]]], 
    return_details: bool = False,
    **kwargs: Any
) -> List[float] | List[Dict[str, Any]]:
    """
    Enhanced reward function that can optionally return detailed feedback.
    
    Args:
        completions: List of model completions (raw output strings)
        expected_outputs: List of expected ground truth grids
        return_details: If True, return detailed feedback dict instead of just scores
        **kwargs: Additional arguments (unused)
        
    Returns:
        List of reward scores, or list of detailed feedback dictionaries if return_details=True
        
    Examples:
        >>> detailed_reward_function(["12;34"], [[[1, 2], [3, 4]]], return_details=True)
        [{'reward': 1.0, 'valid_array': True, 'correct_value': True, 'parsed_grid': [[1, 2], [3, 4]]}]
    """
    results = []
    
    for completion, expected in zip(completions, expected_outputs, strict=False):
        # Initialize result tracking
        result = {
            'reward': -1.0,
            'valid_array': False,
            'correct_value': False,
            'parsed_grid': None,
            'error_message': None
        }
        
        try:
            # Check if the completion can be parsed into a valid array
            if not check_array(completion):
                result['error_message'] = "Invalid array format - cannot be parsed"
            else:
                result['valid_array'] = True
                
                # Try to parse the grid
                parsed_grid = parse_grid_from_string(completion)
                result['parsed_grid'] = parsed_grid
                
                if parsed_grid is None:
                    result['error_message'] = "Failed to parse grid despite passing array check"
                else:
                    # Valid array format, now check if the value is correct
                    if check_value(completion, expected):
                        result['correct_value'] = True
                        result['reward'] = 1.0
                    else:
                        result['reward'] = 0.0
                        result['error_message'] = f"Valid format but incorrect result. Expected: {expected}, Got: {parsed_grid}"
        
        except Exception as e:
            result['error_message'] = f"Unexpected error during evaluation: {str(e)}"
        
        if return_details:
            results.append(result)
        else:
            results.append(result['reward'])
    
    return results


def batch_evaluate_transduction(
    completions: List[str], 
    expected_outputs: List[List[List[int]]]
) -> Dict[str, Any]:
    """
    Evaluate a batch of transduction completions and return summary statistics.
    
    Args:
        completions: List of model completions
        expected_outputs: List of expected ground truth grids
        
    Returns:
        Dictionary with evaluation statistics
        
    Examples:
        >>> batch_evaluate_transduction(["12;34", "invalid", "12;35"], 
        ...                            [[[1,2],[3,4]], [[1,2],[3,4]], [[1,2],[3,4]]])
        {'total': 3, 'correct': 1, 'valid_format': 2, 'invalid_format': 1, 'accuracy': 0.333...}
    """
    detailed_results = detailed_reward_function(
        completions, expected_outputs, return_details=True
    )
    
    total = len(detailed_results)
    correct = sum(1 for r in detailed_results if r['correct_value'])
    valid_format = sum(1 for r in detailed_results if r['valid_array'])
    invalid_format = sum(1 for r in detailed_results if not r['valid_array'])
    
    return {
        'total': total,
        'correct': correct,
        'valid_format': valid_format,
        'invalid_format': invalid_format,
        'accuracy': correct / total if total > 0 else 0.0,
        'valid_format_rate': valid_format / total if total > 0 else 0.0,
        'conditional_accuracy': correct / valid_format if valid_format > 0 else 0.0,
        'mean_reward': sum(r['reward'] for r in detailed_results) / total if total > 0 else 0.0
    }
