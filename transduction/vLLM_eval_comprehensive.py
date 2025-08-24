"""
vLLM-based comprehensive evaluation system for ARC transduction models.

This script provides a vLLM-based version of the comprehensive evaluation system,
offering faster inference through vLLM's optimized serving and batching capabilities
while supporting all inference techniques from the original evaluator.

Features:
- vLLM-based inference for improved speed and throughput
- Support for LoRA adapters (SFT/RL models)
- Batch processing for multiple samples
- All inference techniques: Standard, Multi-sample, AIRV, TTFT, TTFT+AIRV
- Compatible with existing evaluation framework
- Comprehensive analysis and reporting
"""

import json
import os
import sys
import argparse
import random
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from loader import load_evaluation_problem, load_training_problem, list_evaluation_problems, list_training_problems
from transduction.prompts import PROMPT_V1
from transduction.data_gen import grid_to_row_strings, format_train_examples

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available. Please install with: pip install vllm")
    VLLM_AVAILABLE = False

# HuggingFace imports (for tokenizer)
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Please install with: pip install transformers")
    HF_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for a model to be evaluated."""
    name: str                    # Human-readable name
    model_type: str             # "instruct", "sft", "rl"
    model_path: str             # Path to model or HF model name
    base_model: Optional[str] = None  # Base model for LoRA adapters
    lora_path: Optional[str] = None   # Path to LoRA adapter
    description: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for an inference technique."""
    name: str                    # Human-readable name
    technique_class: type       # Class implementing the inference technique
    params: Dict[str, Any]      # Parameters for the technique
    description: Optional[str] = None
    category: str = "standard"  # Category: "standard", "airv", "ttft", "ttft_airv"


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


class vLLMARCTransductionInference:
    """
    vLLM-based ARC transduction inference class for high-performance inference.
    """
    
    def __init__(self, 
                 model_path: str,
                 base_model: Optional[str] = None,
                 lora_path: Optional[str] = None,
                 gpu_memory_utilization: float = 0.8,
                 max_lora_rank: int = 64,
                 tensor_parallel_size: int = 1,
                 **kwargs):
        """
        Initialize vLLM inference.
        
        Args:
            model_path: Path to model (base model if using LoRA)
            base_model: Base model name (for LoRA)
            lora_path: Path to LoRA adapter
            gpu_memory_utilization: GPU memory utilization ratio
            max_lora_rank: Maximum LoRA rank
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for this inference class")
        
        self.model_path = model_path
        self.base_model = base_model
        self.lora_path = lora_path
        
        # Determine the actual model to load
        if lora_path and base_model:
            # Using LoRA adapter
            actual_model_path = base_model
            self.use_lora = True
            print(f"Loading base model: {base_model}")
            print(f"With LoRA adapter: {lora_path}")
        else:
            # Using full model
            actual_model_path = model_path
            self.use_lora = False
            print(f"Loading model: {model_path}")
        
        # Initialize vLLM
        self.llm = LLM(
            model=actual_model_path,
            enable_lora=self.use_lora,
            max_lora_rank=max_lora_rank if self.use_lora else None,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            **kwargs
        )
        
        # Setup LoRA request if needed
        self.lora_request = None
        if self.use_lora and lora_path:
            self.lora_request = LoRARequest(
                lora_name=f"adapter_{Path(lora_path).name}",
                lora_int_id=1,
                lora_path=lora_path
            )
        
        # Load tokenizer for prompt formatting
        tokenizer_path = base_model if base_model else model_path
        if HF_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = None
            print("Warning: HuggingFace transformers not available, tokenizer-based features disabled")
        
        print("vLLM model loaded successfully!")
    
    def format_problem_for_inference(self, problem_data: Dict[str, Any], 
                                   train_sample_count: int = 3,
                                   test_example_idx: int = 0) -> str:
        """
        Format an ARC problem for inference using the transduction prompt.
        """
        # Get training examples
        train_examples = problem_data.get('train', [])
        if len(train_examples) < train_sample_count:
            sampled_train = train_examples
        else:
            sampled_train = train_examples[:train_sample_count]
        
        # Get test example
        test_examples = problem_data.get('test', [])
        if not test_examples:
            raise ValueError("No test examples available in problem")
        
        test_example = test_examples[test_example_idx % len(test_examples)]
        
        # Format the prompt
        train_pairs_formatted = format_train_examples(sampled_train)
        test_input_formatted = grid_to_row_strings(test_example['input'])
        test_input_str = ';'.join(test_input_formatted)
        
        # Create a placeholder with same dimensions as test input
        test_placeholder_rows = ['0' * len(row) for row in test_input_formatted]
        test_placeholder_str = ';'.join(test_placeholder_rows)
        
        prompt = PROMPT_V1.format(
            train_pairs=train_pairs_formatted,
            test_input=test_input_str,
            test_output_placeholder=test_placeholder_str
        )
        
        return prompt
    
    def generate_responses(self, prompts: List[str], 
                          temperature: float = 0.1,
                          max_tokens: int = 1024,
                          top_p: float = 0.9,
                          **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts using vLLM batch processing.
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self.llm.generate(
            prompts, 
            sampling_params,
            lora_request=self.lora_request
        )
        
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a single response."""
        responses = self.generate_responses([prompt], **kwargs)
        return responses[0]
    
    def parse_grid_response(self, response: str) -> Optional[List[List[int]]]:
        """
        Parse the model's response to extract a grid.
        """
        response = response.strip()
        
        # Look for semicolon-separated format first
        if ';' in response:
            import re
            grid_match = re.search(r'[0-9;]+', response)
            if grid_match:
                grid_str = grid_match.group()
                try:
                    rows = grid_str.split(';')
                    grid = []
                    for row in rows:
                        if row.strip():  # Skip empty rows
                            grid_row = [int(char) for char in row if char.isdigit()]
                            if grid_row:  # Only add non-empty rows
                                grid.append(grid_row)
                    
                    if grid:  # Only return if we got a valid grid
                        return grid
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def evaluate_prediction(self, prediction: Optional[List[List[int]]], 
                          ground_truth: List[List[int]]) -> bool:
        """
        Evaluate if a prediction matches the ground truth.
        """
        if prediction is None:
            return False
        return prediction == ground_truth
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False,
                           **generation_kwargs) -> Dict[str, Any]:
        """
        Perform inference on a single ARC problem.
        """
        # Format prompt
        prompt = self.format_problem_for_inference(
            problem_data, train_sample_count, test_example_idx
        )
        
        if verbose:
            print(f"Prompt length: {len(prompt)} characters")
            print("=" * 50)
            print("PROMPT:")
            print(prompt)
            print("=" * 50)
        
        # Generate response
        response = self.generate_response(prompt, **generation_kwargs)
        
        if verbose:
            print("MODEL RESPONSE:")
            print(response)
            print("=" * 50)
        
        # Parse response
        predicted_grid = self.parse_grid_response(response)
        
        # Get ground truth
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        
        # Evaluate
        is_correct = self.evaluate_prediction(predicted_grid, ground_truth)
        
        result = {
            'prompt': prompt,
            'response': response,
            'predicted_grid': predicted_grid,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'train_sample_count': train_sample_count,
            'test_example_idx': test_example_idx
        }
        
        if verbose:
            print(f"GROUND TRUTH:")
            for row in ground_truth:
                print(';'.join(map(str, row)))
            print()
            
            print(f"PREDICTED:")
            if predicted_grid:
                for row in predicted_grid:
                    print(';'.join(map(str, row)))
            else:
                print("Failed to parse prediction")
            print()
            
            print(f"CORRECT: {is_correct}")
            print("=" * 50)
        
        return result


class InferenceTechnique(ABC):
    """Abstract base class for inference techniques."""
    
    @abstractmethod
    def __init__(self, model_config: ModelConfig, **kwargs):
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


class vLLMStandardInference(InferenceTechnique):
    """Standard inference technique using vLLM."""
    
    def __init__(self, model_config: ModelConfig, **kwargs):
        self.model_config = model_config
        self.inference = vLLMARCTransductionInference(
            model_path=model_config.model_path,
            base_model=model_config.base_model,
            lora_path=model_config.lora_path,
            **kwargs
        )
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        return self.inference.infer_single_problem(
            problem_data, train_sample_count, test_example_idx, verbose,
            temperature=0.1, max_tokens=1024
        )
    
    def cleanup(self):
        # vLLM handles cleanup automatically
        pass


class vLLMMultiSampleInference(InferenceTechnique):
    """Multi-sample inference technique using vLLM batch processing."""
    
    def __init__(self, model_config: ModelConfig, num_samples: int = 5, **kwargs):
        self.model_config = model_config
        self.num_samples = num_samples
        self.inference = vLLMARCTransductionInference(
            model_path=model_config.model_path,
            base_model=model_config.base_model,
            lora_path=model_config.lora_path,
            **kwargs
        )
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        # Format prompt once
        prompt = self.inference.format_problem_for_inference(
            problem_data, train_sample_count, test_example_idx
        )
        
        # Generate multiple samples using batch processing
        prompts = [prompt] * self.num_samples
        responses = self.inference.generate_responses(
            prompts, 
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024
        )
        
        # Process all samples
        samples = []
        for i, response in enumerate(responses):
            predicted_grid = self.inference.parse_grid_response(response)
            samples.append({
                'response': response,
                'predicted_grid': predicted_grid,
                'sample_idx': i
            })
        
        # Get ground truth
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        
        # Find the best sample (most commonly present grid among parseable samples)
        grid_counts = {}
        parseable_samples = []
        
        for sample in samples:
            if sample['predicted_grid'] is not None:
                parseable_samples.append(sample)
                # Convert grid to string for counting
                grid_str = ';'.join([''.join(map(str, row)) for row in sample['predicted_grid']])
                grid_counts[grid_str] = grid_counts.get(grid_str, 0) + 1
        
        best_sample = None
        if parseable_samples:
            # Find the most common grid
            most_common_grid = max(grid_counts, key=grid_counts.get)
            # Find the first sample with this grid
            for sample in parseable_samples:
                grid_str = ';'.join([''.join(map(str, row)) for row in sample['predicted_grid']])
                if grid_str == most_common_grid:
                    best_sample = sample
                    break
        
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
        # vLLM handles cleanup automatically
        pass


class vLLMAIRVInference(InferenceTechnique):
    """AIRV inference technique adapted for vLLM."""
    
    def __init__(self, model_config: ModelConfig, 
                 num_augmentations: int = 4,
                 include_original: bool = True,
                 **kwargs):
        self.model_config = model_config
        self.num_augmentations = num_augmentations
        self.include_original = include_original
        self.inference = vLLMARCTransductionInference(
            model_path=model_config.model_path,
            base_model=model_config.base_model,
            lora_path=model_config.lora_path,
            **kwargs
        )
    
    def augment_problem(self, problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create augmented versions of the problem."""
        # Import augmentation functions
        try:
            from augment import augment_problem
            augmented_problems = []
            
            # Add original if requested
            if self.include_original:
                augmented_problems.append(problem_data)
            
            # Create augmented versions
            for _ in range(self.num_augmentations):
                try:
                    aug_problem = augment_problem(problem_data)
                    augmented_problems.append(aug_problem)
                except Exception as e:
                    print(f"Warning: Failed to create augmentation: {e}")
                    # Fall back to original problem
                    augmented_problems.append(problem_data)
            
            return augmented_problems
        except ImportError:
            print("Warning: augment module not available, using original problem only")
            return [problem_data] * (self.num_augmentations + (1 if self.include_original else 0))
    
    def deaugment_prediction(self, prediction: Optional[List[List[int]]], 
                           original_problem: Dict[str, Any],
                           augmented_problem: Dict[str, Any]) -> Optional[List[List[int]]]:
        """Revert augmentation from prediction."""
        if prediction is None:
            return None
        
        try:
            from deaugment import deaugment_grid
            return deaugment_grid(prediction, original_problem, augmented_problem)
        except ImportError:
            print("Warning: deaugment module not available, returning prediction as-is")
            return prediction
    
    def infer_single_problem(self, problem_data: Dict[str, Any], 
                           train_sample_count: int = 3,
                           test_example_idx: int = 0,
                           verbose: bool = False) -> Dict[str, Any]:
        # Create augmented problems
        augmented_problems = self.augment_problem(problem_data)
        
        # Prepare prompts for all augmented problems
        prompts = []
        for aug_problem in augmented_problems:
            prompt = self.inference.format_problem_for_inference(
                aug_problem, train_sample_count, test_example_idx
            )
            prompts.append(prompt)
        
        # Generate responses for all augmented problems using batch processing
        responses = self.inference.generate_responses(
            prompts, 
            temperature=0.1,
            max_tokens=1024
        )
        
        # Process predictions and deaugment
        predictions = []
        for i, (response, aug_problem) in enumerate(zip(responses, augmented_problems)):
            # Parse prediction
            predicted_grid = self.inference.parse_grid_response(response)
            
            # Deaugment if this is an augmented problem
            if i > 0 or not self.include_original:  # Skip deaugmentation for original problem
                predicted_grid = self.deaugment_prediction(
                    predicted_grid, problem_data, aug_problem
                )
            
            predictions.append({
                'response': response,
                'predicted_grid': predicted_grid,
                'augmentation_idx': i,
                'is_original': i == 0 and self.include_original
            })
        
        # Vote on predictions
        grid_counts = {}
        valid_predictions = []
        
        for pred in predictions:
            if pred['predicted_grid'] is not None:
                valid_predictions.append(pred)
                # Convert grid to string for voting
                grid_str = ';'.join([''.join(map(str, row)) for row in pred['predicted_grid']])
                grid_counts[grid_str] = grid_counts.get(grid_str, 0) + 1
        
        # Select best prediction (most votes)
        best_prediction = None
        if valid_predictions:
            most_common_grid = max(grid_counts, key=grid_counts.get)
            for pred in valid_predictions:
                grid_str = ';'.join([''.join(map(str, row)) for row in pred['predicted_grid']])
                if grid_str == most_common_grid:
                    best_prediction = pred
                    break
        
        # Fallback to first prediction
        if best_prediction is None:
            best_prediction = predictions[0]
        
        # Get ground truth and evaluate
        test_example = problem_data['test'][test_example_idx % len(problem_data['test'])]
        ground_truth = test_example['output']
        is_correct = self.inference.evaluate_prediction(best_prediction['predicted_grid'], ground_truth)
        
        result = {
            'prompt': prompts[0],  # Use original prompt
            'response': best_prediction['response'],
            'predicted_grid': best_prediction['predicted_grid'],
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'train_sample_count': train_sample_count,
            'test_example_idx': test_example_idx,
            'all_predictions': predictions,
            'best_augmentation_idx': best_prediction['augmentation_idx'],
            'num_augmentations': len(augmented_problems),
            'grid_votes': grid_counts
        }
        
        if verbose:
            print(f"AIRV: Generated {len(augmented_problems)} augmented problems")
            print(f"Selected prediction from augmentation {best_prediction['augmentation_idx']}")
            print(f"Vote counts: {grid_counts}")
            print(f"GROUND TRUTH:")
            for row in ground_truth:
                print(';'.join(map(str, row)))
            print()
            
            print(f"PREDICTED:")
            if best_prediction['predicted_grid']:
                for row in best_prediction['predicted_grid']:
                    print(';'.join(map(str, row)))
            else:
                print("Failed to parse prediction")
            print()
            
            print(f"CORRECT: {is_correct}")
            print("=" * 50)
        
        return result
    
    def cleanup(self):
        # vLLM handles cleanup automatically
        pass


class vLLMComprehensiveARCEvaluator:
    """
    vLLM-based comprehensive evaluator for ARC transduction inference techniques.
    """
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.model_configs = {}
        self.inference_configs = {}
        self._register_all_techniques()
    
    def _register_all_techniques(self):
        """Register all available vLLM-based inference techniques."""
        # Standard techniques
        self.register_inference_technique(InferenceConfig(
            name="vllm_standard",
            technique_class=vLLMStandardInference,
            params={},
            description="vLLM standard single-sample inference",
            category="standard"
        ))
        
        self.register_inference_technique(InferenceConfig(
            name="vllm_multi_sample_5",
            technique_class=vLLMMultiSampleInference,
            params={"num_samples": 5},
            description="vLLM multi-sample inference with 5 samples",
            category="standard"
        ))
        
        self.register_inference_technique(InferenceConfig(
            name="vllm_multi_sample_10",
            technique_class=vLLMMultiSampleInference,
            params={"num_samples": 10},
            description="vLLM multi-sample inference with 10 samples",
            category="standard"
        ))
        
        self.register_inference_technique(InferenceConfig(
            name="vllm_multi_sample_20",
            technique_class=vLLMMultiSampleInference,
            params={"num_samples": 20},
            description="vLLM multi-sample inference with 20 samples",
            category="standard"
        ))
        
        # Register AIRV techniques
        self._register_airv_techniques()
        
        # Register TTFT techniques
        self._register_ttft_techniques()
        
        # Register TTFT + AIRV techniques
        self._register_ttft_airv_techniques()
        
        print("Registered vLLM-based inference techniques")
    
    def _register_airv_techniques(self):
        """Register AIRV inference techniques."""
        try:
            # Register various AIRV configurations
            airv_configs = [
                {
                    'name': 'vllm_airv_light',
                    'params': {'num_augmentations': 2, 'include_original': True},
                    'description': 'vLLM AIRV with 2 augmentations + original'
                },
                {
                    'name': 'vllm_airv_standard',
                    'params': {'num_augmentations': 4, 'include_original': True},
                    'description': 'vLLM AIRV with 4 augmentations + original'
                },
                {
                    'name': 'vllm_airv_heavy',
                    'params': {'num_augmentations': 8, 'include_original': True},
                    'description': 'vLLM AIRV with 8 augmentations + original'
                },
                {
                    'name': 'vllm_airv_no_original',
                    'params': {'num_augmentations': 5, 'include_original': False},
                    'description': 'vLLM AIRV with 5 augmentations only'
                }
            ]
            
            for config in airv_configs:
                self.register_inference_technique(InferenceConfig(
                    name=config['name'],
                    technique_class=vLLMAIRVInference,
                    params=config['params'],
                    description=config['description'],
                    category="airv"
                ))
            
            print(f"Registered {len(airv_configs)} vLLM AIRV techniques")
            
        except Exception as e:
            print(f"Warning: Could not register AIRV techniques: {e}")
    
    def _register_ttft_techniques(self):
        """Register TTFT inference techniques."""
        print("Note: TTFT techniques require model fine-tuning and are not yet implemented for vLLM")
        print("Consider using the original eval_comprehensive.py for TTFT evaluation")
    
    def _register_ttft_airv_techniques(self):
        """Register TTFT + AIRV combined techniques."""
        print("Note: TTFT + AIRV techniques require model fine-tuning and are not yet implemented for vLLM")
        print("Consider using the original eval_comprehensive.py for TTFT + AIRV evaluation")
    
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
                model_path=base_model,  # Base model path
                base_model=base_model,
                lora_path=sft_path,     # LoRA adapter path
                description=f"SFT model based on {base_model}"
            ))
        
        # RL model (assumes standard path structure)
        rl_path = "qwen2.5_0.5b_arc_transduction_rl/final"
        if Path(rl_path).exists():
            self.register_model(ModelConfig(
                name="rl",
                model_type="rl",
                model_path=base_model,  # Base model path
                base_model=base_model,
                lora_path=rl_path,      # LoRA adapter path
                description=f"RL model based on {base_model}"
            ))
    
    def evaluate_single_combination(self, 
                                  model_config: ModelConfig,
                                  inference_config: InferenceConfig,
                                  problem_ids: List[str],
                                  dataset_type: str = "evaluation",
                                  max_problems: Optional[int] = None,
                                  train_sample_count: int = 3,
                                  verbose: bool = False,
                                  **vllm_kwargs) -> EvaluationResult:
        """Evaluate a single model-technique combination."""
        
        if max_problems:
            problem_ids = problem_ids[:max_problems]
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_config.name} + {inference_config.name}")
        print(f"Model: {model_config.model_path}")
        if model_config.lora_path:
            print(f"LoRA: {model_config.lora_path}")
        print(f"Category: {inference_config.category}")
        print(f"Problems: {len(problem_ids)} from {dataset_type} set")
        print(f"{'='*70}")
        
        # Initialize inference technique
        technique = inference_config.technique_class(
            model_config=model_config,
            **inference_config.params,
            **vllm_kwargs
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
                'max_problems': max_problems,
                'category': inference_config.category,
                'used_vllm': True
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
                                technique_categories: Optional[List[str]] = None,
                                technique_names: Optional[List[str]] = None,
                                dataset_type: str = "evaluation",
                                max_problems: Optional[int] = None,
                                train_sample_count: int = 3,
                                output_file: Optional[str] = None,
                                verbose: bool = False,
                                **vllm_kwargs) -> List[EvaluationResult]:
        """Evaluate all specified model-technique combinations."""
        
        # Get problem IDs
        if dataset_type == "evaluation":
            problem_ids = list_evaluation_problems(self.data_dir)
        else:
            problem_ids = list_training_problems(self.data_dir)
        
        # Shuffle for random sampling
        random.shuffle(problem_ids)
        
        # Determine which models to evaluate
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        # Determine which techniques to evaluate
        if technique_names is None:
            if technique_categories is None:
                technique_names = list(self.inference_configs.keys())
            else:
                # Filter by categories
                technique_names = []
                for name, config in self.inference_configs.items():
                    if config.category in technique_categories:
                        technique_names.append(name)
        
        print(f"Evaluating {len(model_names)} models × {len(technique_names)} techniques")
        print(f"Models: {model_names}")
        print(f"Techniques by category:")
        
        # Group techniques by category for display
        by_category = {}
        for name in technique_names:
            if name in self.inference_configs:
                category = self.inference_configs[name].category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(name)
        
        for category, techniques in by_category.items():
            print(f"  {category}: {techniques}")
        
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
                        verbose=verbose,
                        **vllm_kwargs
                    )
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} + {technique_name}: {e}")
                    continue
        
        # Print comprehensive summary
        self._print_comprehensive_summary(all_results)
        
        # Save results if requested
        if output_file:
            print(f"\nSaving detailed results to {output_file}")
            self.save_results(all_results, output_file)
        
        return all_results
    
    def _print_comprehensive_summary(self, results: List[EvaluationResult]):
        """Print comprehensive summary of all results."""
        print(f"\n{'='*100}")
        print("vLLM COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*100}")
        
        # Overall summary table
        print(f"{'Model':<12} {'Technique':<25} {'Category':<12} {'Accuracy':<10} {'Correct':<8} {'Total':<8} {'Avg Time':<10}")
        print("-" * 100)
        
        for result in results:
            print(f"{result.model_config.name:<12} "
                  f"{result.inference_config.name:<25} "
                  f"{result.inference_config.category:<12} "
                  f"{result.accuracy:<10.3f} "
                  f"{result.correct_predictions:<8} "
                  f"{result.total_problems:<8} "
                  f"{result.avg_inference_time:<10.2f}s")
        
        # Category-based analysis
        print(f"\n{'='*100}")
        print("ANALYSIS BY CATEGORY")
        print(f"{'='*100}")
        
        # Group results by model and category
        model_category_results = {}
        for result in results:
            model_name = result.model_config.name
            category = result.inference_config.category
            
            if model_name not in model_category_results:
                model_category_results[model_name] = {}
            if category not in model_category_results[model_name]:
                model_category_results[model_name][category] = []
            
            model_category_results[model_name][category].append(result)
        
        # Print analysis for each model
        for model_name, category_results in model_category_results.items():
            print(f"\nModel: {model_name}")
            print("-" * 60)
            
            best_by_category = {}
            for category, category_result_list in category_results.items():
                # Find best result in this category
                best_result = max(category_result_list, key=lambda x: x.accuracy)
                best_by_category[category] = best_result
                
                print(f"Best {category:<12}: {best_result.inference_config.name:<25} "
                      f"({best_result.accuracy:.3f} accuracy, {best_result.avg_inference_time:.1f}s)")
            
            # Compare with baseline if available
            if 'standard' in best_by_category:
                baseline = best_by_category['standard']
                print(f"\nComparisons to best standard ({baseline.inference_config.name}):")
                
                for category, best_result in best_by_category.items():
                    if category != 'standard':
                        improvement = best_result.accuracy - baseline.accuracy
                        time_ratio = best_result.avg_inference_time / baseline.avg_inference_time
                        print(f"  {category:<12}: {improvement:+.3f} accuracy, {time_ratio:.1f}x time")
        
        # Overall best results
        print(f"\n{'='*100}")
        print("OVERALL BEST RESULTS")
        print(f"{'='*100}")
        
        if results:
            # Find overall best result
            best_overall = max(results, key=lambda x: x.accuracy)
            print(f"Best Overall: {best_overall.model_config.name} + {best_overall.inference_config.name}")
            print(f"  Accuracy: {best_overall.accuracy:.3f}")
            print(f"  Category: {best_overall.inference_config.category}")
            print(f"  Time: {best_overall.avg_inference_time:.1f}s per problem")
            
            # Best by category across all models
            category_best = {}
            for result in results:
                category = result.inference_config.category
                if category not in category_best or result.accuracy > category_best[category].accuracy:
                    category_best[category] = result
            
            print(f"\nBest by Category (across all models):")
            for category in ['standard', 'airv', 'ttft', 'ttft_airv']:
                if category in category_best:
                    result = category_best[category]
                    print(f"  {category:<12}: {result.model_config.name} + {result.inference_config.name} "
                          f"({result.accuracy:.3f})")
            
            # Speed comparison
            fastest = min(results, key=lambda x: x.avg_inference_time)
            print(f"\nFastest: {fastest.model_config.name} + {fastest.inference_config.name}")
            print(f"  Time: {fastest.avg_inference_time:.1f}s per problem")
            print(f"  Accuracy: {fastest.accuracy:.3f}")
    
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
                    'lora_path': result.model_config.lora_path,
                    'description': result.model_config.description
                },
                'inference_config': {
                    'name': result.inference_config.name,
                    'params': result.inference_config.params,
                    'description': result.inference_config.description,
                    'category': result.inference_config.category
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
    """Main function for running vLLM-based comprehensive ARC evaluation."""
    parser = argparse.ArgumentParser(description='vLLM-based Comprehensive ARC Transduction Evaluation')
    
    # Model configuration
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help='Base model for instruct/SFT/RL variants')
    parser.add_argument('--all_models', action='store_true',
                       help='Evaluate all available models')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model to evaluate (required if --all_models is not set)')
    parser.add_argument('--sft_path', type=str, default="qwen2.5_0.5b_arc_transduction_sft/final",
                       help='Path to SFT model')
    parser.add_argument('--rl_path', type=str, default="qwen2.5_0.5b_arc_transduction_rl/final",
                       help='Path to RL model')
    
    # Inference technique categories
    parser.add_argument('--categories', type=str, nargs='+', 
                       choices=['standard', 'airv', 'ttft', 'ttft_airv', 'all'],
                       default=['standard'],
                       help='Inference technique categories to evaluate')
    parser.add_argument('--techniques', type=str, nargs='+', default=None,
                       help='Specific techniques to evaluate (overrides --categories)')
    
    # Evaluation settings
    parser.add_argument('--data_dir', type=str, default=".",
                       help='Directory containing ARC data')
    parser.add_argument('--dataset', type=str, choices=['evaluation', 'training'], 
                       default='evaluation',
                       help='Dataset to evaluate on')
    parser.add_argument('--max_problems', type=int, default=20,
                       help='Maximum number of problems to evaluate')
    parser.add_argument('--train_samples', type=int, default=3,
                       help='Number of training examples to use per problem')
    
    # vLLM specific settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8,
                       help='GPU memory utilization ratio for vLLM')
    parser.add_argument('--max_lora_rank', type=int, default=64,
                       help='Maximum LoRA rank for vLLM')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    
    # Output and misc
    parser.add_argument('--output', type=str, default="vllm_comprehensive_results.json",
                       help='Output file to save detailed results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate model selection
    if not args.all_models and args.model_name is None:
        parser.error("Either --all_models must be set or --model_name must be specified")
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize evaluator
    evaluator = vLLMComprehensiveARCEvaluator(data_dir=args.data_dir)
    
    # Set up models
    evaluator.setup_default_models(base_model=args.base_model)
    
    # Override default paths if provided
    if args.sft_path != "qwen2.5_0.5b_arc_transduction_sft/final" or Path(args.sft_path).exists():
        evaluator.register_model(ModelConfig(
            name="sft",
            model_type="sft",
            model_path=args.base_model,  # Base model
            base_model=args.base_model,
            lora_path=args.sft_path,     # LoRA adapter
            description=f"Custom SFT model: {args.sft_path}"
        ))
    
    if args.rl_path != "qwen2.5_0.5b_arc_transduction_rl/final" or Path(args.rl_path).exists():
        evaluator.register_model(ModelConfig(
            name="rl",
            model_type="rl",
            model_path=args.base_model,  # Base model
            base_model=args.base_model,
            lora_path=args.rl_path,      # LoRA adapter
            description=f"Custom RL model: {args.rl_path}"
        ))
    
    # Determine which models to evaluate
    if args.all_models:
        model_names = None  # Will use all available models
        print("🚀 Evaluating ALL available models")
    else:
        model_names = [args.model_name]
        print(f"🎯 Evaluating single model: {args.model_name}")
    
    # Determine technique categories
    if 'all' in args.categories:
        technique_categories = ['standard', 'airv']  # TTFT not yet implemented for vLLM
    else:
        technique_categories = args.categories
    
    print(f"📊 Technique categories: {technique_categories}")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🔢 Max problems: {args.max_problems}")
    print(f"🚀 Using vLLM for fast inference")
    
    # Prepare vLLM kwargs
    vllm_kwargs = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_lora_rank': args.max_lora_rank,
        'tensor_parallel_size': args.tensor_parallel_size
    }
    
    # Run comprehensive evaluation
    start_time = time.time()
    
    results = evaluator.evaluate_all_combinations(
        model_names=model_names,
        technique_categories=technique_categories,
        technique_names=args.techniques,
        dataset_type=args.dataset,
        max_problems=args.max_problems,
        train_sample_count=args.train_samples,
        output_file=args.output,
        verbose=args.verbose,
        **vllm_kwargs
    )
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 vLLM Comprehensive evaluation completed!")
    print(f"⏱️ Total time: {total_time:.1f}s")
    print(f"📊 Evaluated {len(results)} model-technique combinations")
    print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
