# vLLM-based ARC Evaluation

This directory contains a high-performance vLLM-based version of the comprehensive ARC evaluation system. vLLM provides significant speed improvements over standard HuggingFace transformers through optimized inference and batching.

## Features

- **Fast Inference**: Uses vLLM for optimized GPU inference
- **LoRA Support**: Full support for SFT and RL models with LoRA adapters
- **Batch Processing**: Efficient multi-sample inference with batching
- **Compatible Interface**: Same command-line interface as the original evaluator
- **Memory Efficient**: Configurable GPU memory utilization

## Installation

```bash
# Install vLLM (requires CUDA)
pip install vllm

# Install other dependencies if not already installed
pip install transformers torch
```

## Quick Start

### Basic Usage

```bash
# Evaluate instruct model with standard techniques
python vLLM_eval_comprehensive.py \
    --model_name instruct \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --categories standard \
    --max_problems 10

# Evaluate all models with multi-sampling
python vLLM_eval_comprehensive.py \
    --all_models \
    --categories standard \
    --max_problems 20 \
    --gpu_memory_utilization 0.8
```

### Using LoRA Models (SFT/RL)

```bash
# Evaluate SFT model
python vLLM_eval_comprehensive.py \
    --model_name sft \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --sft_path "path/to/sft/adapter" \
    --categories standard \
    --max_problems 10

# Evaluate RL model
python vLLM_eval_comprehensive.py \
    --model_name rl \
    --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
    --rl_path "path/to/rl/adapter" \
    --categories standard \
    --max_problems 10
```

### Multi-GPU Setup

```bash
# Use tensor parallelism across multiple GPUs
python vLLM_eval_comprehensive.py \
    --all_models \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --max_problems 50
```

## Command Line Options

### Model Configuration
- `--base_model`: Base model name (default: "Qwen/Qwen2.5-0.5B-Instruct")
- `--all_models`: Evaluate all available models
- `--model_name`: Specific model to evaluate
- `--sft_path`: Path to SFT LoRA adapter
- `--rl_path`: Path to RL LoRA adapter

### Inference Techniques
- `--categories`: Technique categories (`standard`, `all`)
- `--techniques`: Specific techniques to evaluate

### Evaluation Settings
- `--dataset`: Dataset to use (`evaluation`, `training`)
- `--max_problems`: Maximum number of problems to evaluate
- `--train_samples`: Number of training examples per problem

### vLLM Configuration
- `--gpu_memory_utilization`: GPU memory utilization ratio (default: 0.8)
- `--max_lora_rank`: Maximum LoRA rank (default: 64)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)

### Output
- `--output`: Output file for results (default: "vllm_comprehensive_results.json")
- `--verbose`: Verbose output
- `--seed`: Random seed for reproducibility

## Available Inference Techniques

### Standard Techniques
- `vllm_standard`: Single-sample inference with vLLM
- `vllm_multi_sample_5`: Multi-sample inference with 5 samples
- `vllm_multi_sample_10`: Multi-sample inference with 10 samples
- `vllm_multi_sample_20`: Multi-sample inference with 20 samples

## Performance Benefits

vLLM provides several performance advantages:

1. **Faster Inference**: Optimized CUDA kernels and memory management
2. **Batch Processing**: Efficient batching for multi-sample techniques
3. **Memory Efficiency**: Better GPU memory utilization
4. **LoRA Optimization**: Efficient LoRA adapter loading and switching

## Example Results Format

The evaluation produces results in JSON format:

```json
{
  "model_config": {
    "name": "sft",
    "model_type": "sft",
    "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora_path": "path/to/sft/adapter"
  },
  "inference_config": {
    "name": "vllm_multi_sample_5",
    "category": "standard"
  },
  "accuracy": 0.85,
  "avg_inference_time": 1.2,
  "total_problems": 20,
  "correct_predictions": 17
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `--gpu_memory_utilization` or `--max_problems`
2. **LoRA Loading Errors**: Ensure LoRA adapters are in PEFT format with correct paths
3. **Model Loading Issues**: Verify model paths and HuggingFace access

### Memory Requirements

- **Base Model**: ~1-2GB GPU memory
- **LoRA Adapters**: Additional ~100-500MB per adapter
- **Inference**: ~2-4GB depending on batch size and sequence length

## Comparison with Original Evaluator

| Feature | Original | vLLM Version |
|---------|----------|--------------|
| Inference Speed | Baseline | 2-5x faster |
| Memory Usage | Higher | Lower |
| Multi-sample Batching | Sequential | Parallel |
| LoRA Support | Manual loading | Optimized |
| GPU Utilization | Variable | Configurable |

## Integration

The vLLM evaluator is designed to be a drop-in replacement for the original comprehensive evaluator:

```python
# Original usage
from transduction.eval_comprehensive import ComprehensiveARCEvaluator

# vLLM usage  
from vLLM_eval_comprehensive import vLLMComprehensiveARCEvaluator

# Same interface
evaluator = vLLMComprehensiveARCEvaluator()
evaluator.setup_default_models()
results = evaluator.evaluate_all_combinations(...)
```

## Contributing

When adding new inference techniques:

1. Inherit from `InferenceTechnique`
2. Implement required methods: `__init__`, `infer_single_problem`, `cleanup`
3. Register with the evaluator using `register_inference_technique`
4. Consider vLLM-specific optimizations (batching, memory management)

## License

Same as the parent project.
