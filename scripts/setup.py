#!/usr/bin/env python3
"""Setup script for ARC-back project with uv."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def main():
    """Main setup function."""
    print("🚀 Setting up ARC-back project with uv...")
    
    # Check if uv is installed
    try:
        run_command("uv --version")
    except subprocess.CalledProcessError:
        print("❌ uv is not installed. Please install it first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Create virtual environment and install dependencies
    print("\n📦 Creating virtual environment and installing dependencies...")
    run_command("uv sync")
    
    # Install pre-commit hooks
    print("\n🔧 Setting up pre-commit hooks...")
    run_command("uv run pre-commit install", check=False)
    
    # Create necessary directories
    print("\n📁 Creating necessary directories...")
    directories = [
        "tests",
        "tests/transduction",
        "tests/induction",
        "data",
        "models",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("\n✅ Setup complete! You can now:")
    print("   • Activate the environment: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)")
    print("   • Or use uv run: uv run python script.py")
    print("   • Run tests: uv run pytest")
    print("   • Format code: uv run black .")
    print("   • Type check: uv run mypy .")


if __name__ == "__main__":
    main()
