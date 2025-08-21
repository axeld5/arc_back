#!/usr/bin/env python3
"""Training script launcher for ARC-back project."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str) -> None:
    """Run a command and handle errors."""
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    """Main training launcher."""
    parser = argparse.ArgumentParser(description="ARC-back training launcher")
    parser.add_argument(
        "task", 
        choices=["transduction", "induction"], 
        help="Task to train on"
    )
    parser.add_argument(
        "mode", 
        choices=["sft", "rl", "both"], 
        help="Training mode"
    )
    parser.add_argument(
        "--data-dir", 
        default=".", 
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        help="Base model to use"
    )
    parser.add_argument(
        "--max-problems", 
        type=int, 
        help="Maximum number of problems to use for training"
    )
    
    args = parser.parse_args()
    
    # Generate data first if needed
    data_file = f"{args.task}/train_dataset.json"
    if not Path(data_file).exists():
        print(f"📊 Generating training data for {args.task}...")
        cmd = f"uv run python {args.task}/data_gen.py --output {data_file}"
        if args.max_problems:
            cmd += f" --max_problems {args.max_problems}"
        run_command(cmd)
    
    # Run SFT training
    if args.mode in ["sft", "both"]:
        print(f"🎯 Starting SFT training for {args.task}...")
        run_command(f"uv run python {args.task}/training/sft.py")
    
    # Run RL training
    if args.mode in ["rl", "both"]:
        print(f"🎮 Starting RL training for {args.task}...")
        run_command(f"uv run python {args.task}/training/rl.py")
    
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
