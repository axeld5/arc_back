#!/usr/bin/env python3
"""
Simple usage example for vLLM_eval_comprehensive.py

This script demonstrates how to use the vLLM-based evaluation system.
"""

import subprocess
import sys
from pathlib import Path

def run_vllm_evaluation():
    """Run a simple vLLM evaluation example."""
    
    # Check if the evaluation script exists
    script_path = Path("vLLM_eval_comprehensive.py")
    if not script_path.exists():
        print("Error: vLLM_eval_comprehensive.py not found!")
        return
    
    # Example command for evaluating a single model with standard techniques
    cmd = [
        sys.executable, "vLLM_eval_comprehensive.py",
        "--model_name", "instruct",  # Evaluate just the instruct model
        "--base_model", "Qwen/Qwen2.5-0.5B-Instruct",  # Base model to use
        "--categories", "standard",  # Use standard inference techniques
        "--max_problems", "5",  # Test on 5 problems for quick evaluation
        "--dataset", "evaluation",  # Use evaluation dataset
        "--gpu_memory_utilization", "0.6",  # Leave some GPU memory free
        "--output", "vllm_test_results.json",  # Output file
        "--verbose"  # Verbose output
    ]
    
    print("Running vLLM evaluation with command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Evaluation failed!")
            print("\nError output:")
            print(result.stderr)
            print("\nStandard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")

def show_help():
    """Show available command line options."""
    cmd = [sys.executable, "vLLM_eval_comprehensive.py", "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Available command line options:")
        print("=" * 50)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting help: {e}")

if __name__ == "__main__":
    print("vLLM Evaluation Example")
    print("=" * 30)
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        print("This will run a quick evaluation using vLLM...")
        print("Make sure you have:")
        print("1. vLLM installed: pip install vllm")
        print("2. Sufficient GPU memory available")
        print("3. ARC evaluation data in the current directory")
        print()
        
        response = input("Continue? (y/N): ")
        if response.lower() == 'y':
            run_vllm_evaluation()
        else:
            print("Evaluation cancelled.")
            print("Run with --help to see all available options.")
