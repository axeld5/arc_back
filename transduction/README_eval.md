# ARC Transduction Evaluation System

This evaluation system provides a flexible framework for evaluating different model types (instruct, SFT, RL) and inference techniques on ARC problems.

## Features

- **Multiple Model Types**: Evaluate instruct models, SFT-trained models, and RL-trained models
- **Extensible Inference Techniques**: Plugin architecture for different inference approaches
- **Comprehensive Metrics**: Accuracy, inference time, and detailed per-problem results
- **Batch Evaluation**: Compare multiple model-technique combinations
- **Advanced Techniques**: Chain-of-thought, self-consistency, temperature sweeping, and more

## Quick Start

### Basic Evaluation

```bash
# Evaluate all available models with standard inference
python eval.py --max_problems 10

# Evaluate specific models
python eval.py --models instruct sft --max_problems 20

# Evaluate with specific techniques
python eval.py --models instruct --techniques standard multi_sample_5 --max_problems 15
```

### Advanced Evaluation

```bash
# Run advanced techniques (requires more compute time)
python eval_advanced.py --models instruct --max_problems 10

# Compare all techniques on SFT model
python eval_advanced.py --models sft --techniques standard chain_of_thought self_consistency_7 temperature_sweep
```

## Model Types

### 1. Instruct Models
Base instruction-following models (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)

### 2. SFT Models  
Supervised fine-tuned models trained on ARC data
- Default path: `qwen2.5_0.5b_arc_transduction_sft/final`
- Custom path: `--sft_path your/path/here`

### 3. RL Models
Reinforcement learning trained models
- Default path: `qwen2.5_0.5b_arc_transduction_rl/final`  
- Custom path: `--rl_path your/path/here`

## Inference Techniques

### Built-in Techniques

1. **Standard**: Single-sample generation with low temperature
2. **Multi-Sample**: Generate multiple samples, pick first correct/parseable one
3. **Chain-of-Thought**: Encourage step-by-step reasoning before final answer
4. **Self-Consistency**: Generate multiple samples and use majority voting
5. **Temperature Sweep**: Try different temperatures and pick best result

### Adding Custom Techniques

Create a new class inheriting from `InferenceTechnique`:

```python
from transduction.eval import InferenceTechnique

class MyCustomInference(InferenceTechnique):
    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        # Initialize your technique
        pass
    
    def infer_single_problem(self, problem_data, train_sample_count=3, 
                           test_example_idx=0, verbose=False):
        # Implement your inference logic
        return {
            'prompt': prompt,
            'response': response, 
            'predicted_grid': predicted_grid,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            # ... other required fields
        }
    
    def cleanup(self):
        # Clean up resources
        pass

# Register the technique
evaluator.register_inference_technique(InferenceConfig(
    name="my_custom",
    technique_class=MyCustomInference,
    params={"param1": "value1"},
    description="My custom inference technique"
))
```

## Configuration Options

### Command Line Arguments

- `--models`: Models to evaluate (`instruct`, `sft`, `rl`)
- `--techniques`: Inference techniques to use
- `--max_problems`: Maximum number of problems to evaluate
- `--dataset`: Dataset to use (`evaluation` or `training`)
- `--data_dir`: Directory containing ARC data
- `--output`: Output file for detailed results
- `--verbose`: Enable verbose output
- `--seed`: Random seed for reproducibility

### Model Paths

- `--base_model`: Base model for instruct variant
- `--sft_path`: Path to SFT model
- `--rl_path`: Path to RL model

## Output Format

Results are saved as JSON with the following structure:

```json
[
  {
    "model_config": {
      "name": "instruct",
      "model_type": "instruct", 
      "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
      "description": "..."
    },
    "inference_config": {
      "name": "standard",
      "params": {},
      "description": "..."
    },
    "problem_results": [...],
    "total_problems": 20,
    "correct_predictions": 12,
    "accuracy": 0.6,
    "avg_inference_time": 2.5,
    "total_time": 65.3,
    "metadata": {...}
  }
]
```

## Examples

### Evaluate All Model Types

```python
from transduction.eval import ARCEvaluator

evaluator = ARCEvaluator()
evaluator.setup_default_models()

results = evaluator.evaluate_all_combinations(
    model_names=['instruct', 'sft', 'rl'],
    technique_names=['standard', 'multi_sample_5'],
    max_problems=50,
    output_file='comparison_results.json'
)
```

### Custom Model Configuration

```python
from transduction.eval import ARCEvaluator, ModelConfig

evaluator = ARCEvaluator()

# Add custom model
evaluator.register_model(ModelConfig(
    name="my_custom_model",
    model_type="sft",
    model_path="/path/to/my/model", 
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    description="My custom fine-tuned model"
))

# Evaluate
results = evaluator.evaluate_all_combinations(
    model_names=['my_custom_model'],
    max_problems=20
)
```

## Performance Tips

1. **Start Small**: Use `--max_problems 5` for quick tests
2. **GPU Memory**: Advanced techniques may require more VRAM
3. **Batch Size**: Techniques like self-consistency generate multiple samples
4. **Time Estimates**: 
   - Standard: ~2-3s per problem
   - Multi-sample: ~10-15s per problem  
   - Self-consistency: ~30-60s per problem
   - Temperature sweep: ~15-25s per problem

## Troubleshooting

### Model Not Found
- Ensure model paths exist and are accessible
- Check HuggingFace model names are correct
- Verify LoRA adapters are properly saved

### Out of Memory
- Reduce `--max_problems`
- Use smaller models
- Try `--device cpu` for CPU inference
- Reduce number of samples in multi-sample techniques

### Import Errors
- Ensure `transformers` and `torch` are installed
- Check Python path includes parent directory
- Verify all required dependencies are available

## Contributing

To add new inference techniques:

1. Create a class inheriting from `InferenceTechnique`
2. Implement required methods: `__init__`, `infer_single_problem`, `cleanup`
3. Add to `techniques.py` or create a new module
4. Register with the evaluator using `InferenceConfig`
5. Test with a small number of problems first
