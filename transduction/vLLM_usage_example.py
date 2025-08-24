#!/usr/bin/env python3
"""
Comprehensive usage examples for vLLM_eval_comprehensive.py

This script demonstrates various ways to use the vLLM-based evaluation system
with different models, techniques, and configurations.
"""

import subprocess
import sys
import json
from pathlib import Path

def check_requirements():
    """Check if required dependencies are available."""
    print("Checking requirements...")
    
    # Check if vLLM evaluation script exists
    script_path = Path("vLLM_eval_comprehensive.py")
    if not script_path.exists():
        print("❌ Error: vLLM_eval_comprehensive.py not found!")
        return False
    
    # Check if evaluation data exists
    eval_dir = Path("evaluation_data")
    if not eval_dir.exists():
        print("❌ Error: evaluation_data directory not found!")
        print("Make sure ARC evaluation data is available in the current directory")
        return False
    
    print("✅ Requirements check passed")
    return True

def run_basic_evaluation():
    """Run a basic evaluation example."""
    print("\n" + "="*60)
    print("BASIC EVALUATION EXAMPLE")
    print("="*60)
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "instruct",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--categories", "standard",
        "--max_problems", "5",
        "--gpu_memory_utilization", "0.6",
        "--output", "basic_results.json",
        "--verbose"
    ]
    
    print("Running basic evaluation:")
    print(" ".join(cmd))
    return run_command(cmd)

def run_multi_sample_evaluation():
    """Run multi-sample inference evaluation."""
    print("\n" + "="*60)
    print("MULTI-SAMPLE EVALUATION EXAMPLE")
    print("="*60)
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "instruct",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--techniques", "vllm_multi_sample_5", "vllm_multi_sample_10",
        "--max_problems", "10",
        "--gpu_memory_utilization", "0.7",
        "--output", "multi_sample_results.json"
    ]
    
    print("Running multi-sample evaluation:")
    print(" ".join(cmd))
    return run_command(cmd)

def run_airv_evaluation():
    """Run AIRV (Augment, Infer, Revert, Vote) evaluation."""
    print("\n" + "="*60)
    print("AIRV EVALUATION EXAMPLE")
    print("="*60)
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "instruct",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--categories", "airv",
        "--max_problems", "8",
        "--gpu_memory_utilization", "0.8",
        "--output", "airv_results.json"
    ]
    
    print("Running AIRV evaluation:")
    print("Note: This requires augment.py and deaugment.py modules")
    print(" ".join(cmd))
    return run_command(cmd)

def run_lora_evaluation():
    """Run evaluation with LoRA adapters (SFT/RL models)."""
    print("\n" + "="*60)
    print("LoRA ADAPTER EVALUATION EXAMPLE")
    print("="*60)
    
    # Check if SFT path exists
    sft_path = Path("qwen2.5_0.5b_arc_transduction_sft/final")
    if not sft_path.exists():
        print("⚠️  SFT model path not found, using mock path for demonstration")
        sft_path = "path/to/your/sft/adapter"
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "sft",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--sft_path", str(sft_path),
        "--techniques", "vllm_standard", "vllm_multi_sample_5",
        "--max_problems", "10",
        "--gpu_memory_utilization", "0.7",
        "--max_lora_rank", "64",
        "--output", "lora_results.json"
    ]
    
    print("Running LoRA adapter evaluation:")
    print(" ".join(cmd))
    return run_command(cmd)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with all models and techniques."""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION EXAMPLE")
    print("="*60)
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--all_models",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--categories", "standard", "airv",
        "--max_problems", "15",
        "--gpu_memory_utilization", "0.8",
        "--output", "comprehensive_results.json",
        "--seed", "42"
    ]
    
    print("Running comprehensive evaluation:")
    print("This will evaluate all available models with multiple techniques")
    print(" ".join(cmd))
    return run_command(cmd)

def run_multi_gpu_evaluation():
    """Run evaluation with multi-GPU tensor parallelism."""
    print("\n" + "="*60)
    print("MULTI-GPU EVALUATION EXAMPLE")
    print("="*60)
    
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "instruct",
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--techniques", "vllm_multi_sample_20",
        "--max_problems", "20",
        "--tensor_parallel_size", "2",  # Use 2 GPUs
        "--gpu_memory_utilization", "0.9",
        "--output", "multi_gpu_results.json"
    ]
    
    print("Running multi-GPU evaluation:")
    print("Note: This requires multiple GPUs available")
    print(" ".join(cmd))
    return run_command(cmd)

def run_command(cmd, timeout=600):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            print("\nKey output lines:")
            lines = result.stdout.split('\n')
            for line in lines[-20:]:  # Show last 20 lines
                if line.strip() and ('accuracy' in line.lower() or 'results' in line.lower() or '✓' in line or '✗' in line):
                    print(f"  {line}")
            return True
        else:
            print("❌ Evaluation failed!")
            print("\nError output:")
            print(result.stderr[:1000])  # Show first 1000 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Evaluation timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False

def show_results_summary(results_file):
    """Show a summary of evaluation results."""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📊 RESULTS SUMMARY from {results_file}")
        print("="*50)
        
        for result in results:
            model_name = result['model_config']['name']
            technique_name = result['inference_config']['name']
            accuracy = result['accuracy']
            avg_time = result['avg_inference_time']
            
            print(f"{model_name} + {technique_name}:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Avg time: {avg_time:.2f}s")
            print()
            
    except Exception as e:
        print(f"Error reading results: {e}")

def interactive_menu():
    """Show interactive menu for different evaluation examples."""
    while True:
        print("\n" + "="*60)
        print("vLLM EVALUATION EXAMPLES MENU")
        print("="*60)
        print("1. Basic evaluation (instruct model, standard techniques)")
        print("2. Multi-sample evaluation (5 and 10 samples)")
        print("3. AIRV evaluation (augmentation-based)")
        print("4. LoRA adapter evaluation (SFT model)")
        print("5. Comprehensive evaluation (all models & techniques)")
        print("6. Multi-GPU evaluation (tensor parallelism)")
        print("7. Show help for all options")
        print("8. View results summary")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            run_basic_evaluation()
        elif choice == "2":
            run_multi_sample_evaluation()
        elif choice == "3":
            run_airv_evaluation()
        elif choice == "4":
            run_lora_evaluation()
        elif choice == "5":
            run_comprehensive_evaluation()
        elif choice == "6":
            run_multi_gpu_evaluation()
        elif choice == "7":
            show_help()
        elif choice == "8":
            results_file = input("Enter results file name (e.g., basic_results.json): ").strip()
            if results_file:
                show_results_summary(results_file)
        else:
            print("Invalid choice. Please try again.")

def show_help():
    """Show detailed help information."""
    cmd = [sys.executable, "vLLM_eval_comprehensive.py", "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\n" + "="*60)
        print("COMMAND LINE OPTIONS")
        print("="*60)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting help: {e}")

def main():
    """Main function."""
    print("vLLM Comprehensive Evaluation Examples")
    print("="*40)
    
    if not check_requirements():
        print("\nPlease ensure:")
        print("1. vLLM is installed: pip install vllm")
        print("2. vLLM_eval_comprehensive.py is in the current directory")
        print("3. ARC evaluation data is available")
        print("4. Sufficient GPU memory is available")
        return
    
    print("\nThis script provides examples of using the vLLM evaluation system.")
    print("Each example demonstrates different features and configurations.")
    
    if len(sys.argv) > 1:
        # Non-interactive mode
        example = sys.argv[1].lower()
        if example == "basic":
            run_basic_evaluation()
        elif example == "multi":
            run_multi_sample_evaluation()
        elif example == "airv":
            run_airv_evaluation()
        elif example == "lora":
            run_lora_evaluation()
        elif example == "comprehensive":
            run_comprehensive_evaluation()
        elif example == "multi-gpu":
            run_multi_gpu_evaluation()
        else:
            print(f"Unknown example: {example}")
            print("Available examples: basic, multi, airv, lora, comprehensive, multi-gpu")
    else:
        # Interactive mode
        interactive_menu()

if __name__ == "__main__":
    main()
