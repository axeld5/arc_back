"""
Repeat-Placeholder Inference for ARC Transduction

Idea: Run inference once to get a predicted TEST OUTPUT. Then rewrite the prompt
so that TEST OUTPUT PLACEHOLDER equals the model's predicted output, and run
inference again. Optionally repeat for multiple passes and pick the final/most
consistent prediction.
"""

import os
import sys
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transduction.inference.inference import ARCTransductionInference
from transduction.data_gen import grid_to_row_strings, format_train_examples
from transduction.prompts import PROMPT_V1


class RepeatPlaceholderInference:
    """
    Two-pass (or multi-pass) inference where the first prediction is fed back as
    the placeholder in the prompt for a subsequent pass.
    """

    def __init__(self, model_name: str, device: str = "auto", num_passes: int = 2, temperature: float = 0.1, **kwargs):
        self.inference = ARCTransductionInference(model_name=model_name, device=device)
        self.num_passes = max(1, int(num_passes))
        # Configure generation
        self.inference.generation_config.temperature = temperature
        self.inference.generation_config.do_sample = temperature > 0.0

    def _format_prompt_with_placeholder(self, problem_data: Dict[str, Any], placeholder_rows: str, train_sample_count: int = 3, test_example_idx: int = 0) -> str:
        train_examples = problem_data.get('train', [])
        if len(train_examples) < train_sample_count:
            sampled_train = train_examples
        else:
            sampled_train = train_examples[:train_sample_count]

        test_examples = problem_data.get('test', [])
        if not test_examples:
            raise ValueError("No test examples available in problem")

        test_example = test_examples[test_example_idx % len(test_examples)]

        train_pairs_formatted = format_train_examples(sampled_train)
        test_input_formatted = grid_to_row_strings(test_example['input'])
        test_input_str = ';'.join(test_input_formatted)

        prompt = PROMPT_V1.format(
            train_pairs=train_pairs_formatted,
            test_input=test_input_str,
            test_output_placeholder=placeholder_rows
        )
        return prompt

    def _grid_to_placeholder_str(self, grid: Optional[List[List[int]]], reference_input: List[List[int]]) -> str:
        # If no grid, return zeros with input dimensions
        if grid is None:
            rows = len(reference_input)
            cols = len(reference_input[0]) if rows > 0 else 0
            return ';'.join(['0' * cols for _ in range(rows)])
        # Convert grid to semicolon-separated placeholder
        return ';'.join([''.join(str(c) for c in row) for row in grid])

    def infer_single_problem(self, problem_data: Dict[str, Any], train_sample_count: int = 3, test_example_idx: int = 0, verbose: bool = False) -> Dict[str, Any]:
        # First pass: standard prompt (zeros placeholder as per ARCTransductionInference)
        first_result = self.inference.infer_single_problem(
            problem_data,
            train_sample_count=train_sample_count,
            test_example_idx=test_example_idx,
            verbose=False
        )

        all_results = [first_result]

        # Prepare placeholder for next passes
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        next_placeholder = self._grid_to_placeholder_str(first_result['predicted_grid'], test_example['input'])

        # Additional passes
        for _ in range(self.num_passes - 1):
            prompt = self._format_prompt_with_placeholder(problem_data, next_placeholder, train_sample_count, test_example_idx)
            response = self.inference.generate_response(prompt)
            predicted_grid = self.inference.parse_grid_response(response)

            result = {
                'prompt': prompt,
                'response': response,
                'predicted_grid': predicted_grid,
                'ground_truth': test_example['output'],
                'train_sample_count': train_sample_count,
                'test_example_idx': test_example_idx
            }
            result['is_correct'] = self.inference.evaluate_prediction(predicted_grid, test_example['output'])
            all_results.append(result)

            # Update placeholder for next iteration
            next_placeholder = self._grid_to_placeholder_str(predicted_grid, test_example['input'])

        final = all_results[-1]
        final['all_pass_results'] = all_results
        final['inference_method'] = 'repeat_placeholder'
        if verbose:
            print(f"Repeat passes: {self.num_passes}, final correct: {final['is_correct']}")
        return final

    def cleanup(self):
        if hasattr(self.inference, 'model'):
            del self.inference.model
        if hasattr(self.inference, 'tokenizer'):
            del self.inference.tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


