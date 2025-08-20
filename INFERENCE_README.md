# ARC Transduction Inference

This program performs inference on ARC (Abstraction and Reasoning Corpus) problems using the transduction approach with language models. It directly applies prompts to problems without using pre-generated datasets.

## Features

- **Direct Inference**: Uses prompts directly on ARC problems without pre-generated datasets
- **Qwen-2.5-0.5B-Instruct Support**: Tested with Qwen-2.5-0.5B-Instruct model from HuggingFace
- **Flexible Problem Loading**: Can work with both evaluation and training datasets
- **Comprehensive Evaluation**: Includes accuracy metrics and detailed result analysis
- **Grid Parsing**: Robust parsing of model outputs back to grid format

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the ARC dataset files in the correct structure:
```
arc_back/
├── evaluation/
│   ├── 00576224.json
│   ├── 009d5c81.json
│   └── ... (more evaluation problems)
├── training/
│   ├── 007bbfb7.json
│   ├── 00d62c1b.json
│   └── ... (more training problems)
└── ...
```

## Usage

### Basic Usage

Test on a single problem:
```bash
python transduction/inference.py --problem_id 00576224
```

Test on multiple evaluation problems:
```bash
python transduction/inference.py --max_problems 5
```

Test on training problems:
```bash
python transduction/inference.py --dataset training --max_problems 5
```

### Advanced Usage

Use a different model:
```bash
python transduction/inference.py --model "microsoft/DialoGPT-medium" --max_problems 3
```

Save results to file:
```bash
python transduction/inference.py --max_problems 10 --output results.json
```

Use more training examples per problem:
```bash
python transduction/inference.py --train_samples 4 --max_problems 5
```

Force CPU usage:
```bash
python transduction/inference.py --device cpu --max_problems 3
```

### Command Line Arguments

- `--model`: HuggingFace model name (default: "Qwen/Qwen2.5-0.5B-Instruct")
- `--device`: Device to use ("auto", "cpu", "cuda")
- `--data_dir`: Directory containing ARC data (default: ".")
- `--dataset`: Dataset to use ("evaluation" or "training")
- `--problem_id`: Test specific problem ID
- `--max_problems`: Maximum number of problems to test
- `--train_samples`: Number of training examples to use per problem
- `--output`: Save results to JSON file
- `--seed`: Random seed for reproducibility

## Quick Test

Run the test script to verify everything works:
```bash
python test_inference.py
```

This will:
1. Load the Qwen-2.5-0.5B-Instruct model
2. Test inference on a single problem with detailed output
3. Optionally test on multiple problems

## How It Works

1. **Problem Loading**: Loads ARC problems from JSON files
2. **Prompt Formatting**: Formats problems using the transduction prompt template
3. **Model Inference**: Generates responses using the language model
4. **Output Parsing**: Parses model responses back to grid format
5. **Evaluation**: Compares predictions with ground truth

### Prompt Template

The program uses this prompt template:
```
Task: TRANSDUCTION for ARC. From TRAIN x→y pairs, directly predict TEST.y for TEST.x by pattern imitation.
Think silently. Do NOT explain. Output ONLY the final grid, correcting the test output placeholder.

Grid encoding:
- Digits 0-9.
- Grid rows separated by semicolons, e.g. "012;340"

TRAIN INPUT 1: [input_grid]
TRAIN OUTPUT 1: [output_grid]

TRAIN INPUT 2: [input_grid]
TRAIN OUTPUT 2: [output_grid]

TEST INPUT: [test_input]
TEST OUTPUT PLACEHOLDER: [placeholder]
TEST OUTPUT: 
```

## Example Output

```
Problem 1/5: 00576224
Prompt length: 245 characters
==================================================
PROMPT:
Task: TRANSDUCTION for ARC. From TRAIN x→y pairs, directly predict TEST.y for TEST.x by pattern imitation.
Think silently. Do NOT explain. Output ONLY the final grid, correcting the test output placeholder.

Grid encoding:
- Digits 0-9.
- Grid rows separated by semicolons, e.g. "012;340"

TRAIN INPUT 1: 86;64
TRAIN OUTPUT 1: 868686;646464;686868;464646;868686;646464

TRAIN INPUT 2: 79;43
TRAIN OUTPUT 2: 797979;434343;979797;343434;797979;434343

TEST INPUT: 32;78
TEST OUTPUT PLACEHOLDER: 00;00
TEST OUTPUT: 
==================================================
MODEL RESPONSE:
323232;787878;232323;878787;323232;787878
==================================================
GROUND TRUTH:
3;2;3;2;3;2
2;7;8;7;8;7;8
7;8;7;8;7;8

PREDICTED:
3;2;3;2;3;2
7;8;7;8;7;8
2;3;2;3;2;3
8;7;8;7;8;7
3;2;3;2;3;2
7;8;7;8;7;8

CORRECT: True
==================================================
✓ Correct

Evaluation Results:
Total problems: 5
Correct predictions: 3
Accuracy: 0.600
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use `--device cpu` or a smaller model
2. **Model Loading Error**: Check internet connection and HuggingFace access
3. **No Problems Found**: Verify ARC dataset files are in correct locations
4. **Parsing Errors**: Model output format may vary; check response parsing logic

### Performance Tips

- Use GPU for faster inference (`--device cuda`)
- Reduce `--train_samples` for faster processing
- Use `--max_problems` to limit evaluation scope
- Consider model quantization for memory efficiency

## Files

- `transduction/inference.py`: Main inference program
- `transduction/prompts.py`: Prompt templates
- `test_inference.py`: Test script
- `requirements.txt`: Python dependencies
- `loader.py`: ARC problem loading utilities

## Model Support

While tested with Qwen-2.5-0.5B-Instruct, the program should work with most HuggingFace causal language models. Some models may require adjustments to the generation configuration or prompt format.
