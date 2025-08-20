"""
Metrics for evaluating transduction model outputs.

This module provides metrics to evaluate whether model outputs can be parsed 
into valid arrays and whether they match expected values.
"""

import re
from typing import List, Optional


def check_array(output_string: str) -> bool:
    """
    Check if the output string of a model can be turned into a valid array.
    
    This function attempts to parse a model's output string into a valid 2D grid
    following the transduction format (semicolon-separated rows, digits 0-9).
    
    Args:
        output_string: Raw model output string
        
    Returns:
        True if the string can be parsed into a valid array, False otherwise
        
    Examples:
        >>> check_array("12;34;56")
        True
        >>> check_array("1a;23")
        False
        >>> check_array("12;34;")
        False
        >>> check_array("")
        False
        >>> check_array("not a grid")
        False
    """
    if not output_string or not isinstance(output_string, str):
        return False
    
    # Clean the response - remove extra whitespace
    response = output_string.strip()
    
    if not response:
        return False
    
    # Look for semicolon-separated format first
    if ';' in response:
        # Extract the part that looks like a grid (digits and semicolons)
        grid_match = re.search(r'[0-9;]+', response)
        if not grid_match:
            return False
        
        grid_str = grid_match.group()
        
        try:
            rows = grid_str.split(';')
            if not rows:
                return False
            
            grid = []
            expected_width = None
            
            for row in rows:
                if not row.strip():  # Skip empty rows
                    return False
                
                # Check if row contains only digits
                if not row.isdigit():
                    return False
                
                grid_row = [int(char) for char in row]
                
                # Check that all digits are valid (0-9)
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return False
                
                # Check consistent width
                if expected_width is None:
                    expected_width = len(grid_row)
                elif len(grid_row) != expected_width:
                    return False
                
                grid.append(grid_row)
            
            # Must have at least one row with at least one element
            return len(grid) > 0 and len(grid[0]) > 0
            
        except (ValueError, IndexError):
            return False
    
    # If no semicolons, check if it's a single row of digits
    if response.isdigit() and len(response) > 0:
        # Single row case
        return all(0 <= int(char) <= 9 for char in response)
    
    return False


def check_value(output_string: str, expected_value: List[List[int]]) -> bool:
    """
    Check if the returned value matches the expected ground truth.
    
    This function parses the model output and compares it against the expected
    ground truth grid. It returns True only if the parsed output exactly matches
    the expected value.
    
    Args:
        output_string: Raw model output string
        expected_value: Expected ground truth as list of lists of integers
        
    Returns:
        True if the parsed output matches expected_value exactly, False otherwise
        
    Examples:
        >>> check_value("12;34", [[1, 2], [3, 4]])
        True
        >>> check_value("12;35", [[1, 2], [3, 4]])
        False
        >>> check_value("invalid", [[1, 2]])
        False
    """
    if not isinstance(expected_value, list) or not expected_value:
        return False
    
    # First check if we can parse the array
    if not check_array(output_string):
        return False
    
    # Parse the output string into a grid
    parsed_grid = parse_grid_from_string(output_string)
    
    if parsed_grid is None:
        return False
    
    # Compare with expected value
    return parsed_grid == expected_value


def parse_grid_from_string(output_string: str) -> Optional[List[List[int]]]:
    """
    Parse a model output string into a 2D grid.
    
    This is a helper function that extracts and parses the grid from a model's
    raw output string. It follows the same parsing logic as check_array but
    returns the actual parsed grid.
    
    Args:
        output_string: Raw model output string
        
    Returns:
        Parsed grid as list of lists of integers, or None if parsing fails
        
    Examples:
        >>> parse_grid_from_string("12;34")
        [[1, 2], [3, 4]]
        >>> parse_grid_from_string("invalid")
        None
    """
    if not output_string or not isinstance(output_string, str):
        return None
    
    response = output_string.strip()
    
    if not response:
        return None
    
    # Look for semicolon-separated format first
    if ';' in response:
        # Extract the part that looks like a grid (digits and semicolons)
        grid_match = re.search(r'[0-9;]+', response)
        if not grid_match:
            return None
        
        grid_str = grid_match.group()
        
        try:
            rows = grid_str.split(';')
            grid = []
            
            for row in rows:
                if not row.strip():  # Skip empty rows
                    continue
                
                if not row.isdigit():
                    return None
                
                grid_row = [int(char) for char in row]
                
                # Check that all digits are valid (0-9)
                if any(digit < 0 or digit > 9 for digit in grid_row):
                    return None
                
                grid.append(grid_row)
            
            return grid if grid else None
            
        except (ValueError, IndexError):
            return None
    
    # If no semicolons, treat as single row
    if response.isdigit() and len(response) > 0:
        try:
            grid_row = [int(char) for char in response]
            if all(0 <= digit <= 9 for digit in grid_row):
                return [grid_row]
        except ValueError:
            pass
    
    return None
