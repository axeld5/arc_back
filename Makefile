# ARC-back Makefile for common tasks

.PHONY: help setup install test lint format type-check clean train-transduction train-induction

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup the project with uv
	@echo "🚀 Setting up ARC-back project..."
	uv sync
	uv run pre-commit install || true
	@echo "✅ Setup complete!"

install: ## Install dependencies
	uv sync

dev-install: ## Install with development dependencies
	uv sync --all-extras

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=transduction --cov=induction --cov-report=html --cov-report=term

lint: ## Run linting
	uv run flake8 transduction induction scripts
	uv run isort --check-only transduction induction scripts
	uv run black --check transduction induction scripts

format: ## Format code
	uv run isort transduction induction scripts
	uv run black transduction induction scripts

type-check: ## Run type checking
	uv run mypy transduction induction

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/

# Training commands
generate-data: ## Generate training data for both tasks
	uv run python transduction/data_gen.py --output transduction/train_dataset.json
	uv run python induction/data_gen.py --output induction/train_dataset.json

train-transduction-sft: ## Train transduction model with SFT
	uv run python transduction/training/sft.py

train-transduction-rl: ## Train transduction model with RL
	uv run python transduction/training/rl.py

train-transduction: ## Train transduction model (SFT + RL)
	$(MAKE) train-transduction-sft
	$(MAKE) train-transduction-rl

train-induction-sft: ## Train induction model with SFT
	uv run python induction/training/sft.py

train-induction-rl: ## Train induction model with RL
	uv run python induction/training/rl.py

train-induction: ## Train induction model (SFT + RL)
	$(MAKE) train-induction-sft
	$(MAKE) train-induction-rl

train-all: ## Train all models
	$(MAKE) generate-data
	$(MAKE) train-transduction
	$(MAKE) train-induction

# Evaluation commands
eval-transduction: ## Evaluate transduction model
	uv run python transduction/eval.py

eval-induction: ## Evaluate induction model
	uv run python induction/eval.py

# Development commands
pre-commit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

update-deps: ## Update dependencies
	uv lock --upgrade

check: ## Run all checks (lint, type-check, test)
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

# Docker commands (if needed)
docker-build: ## Build Docker image
	docker build -t arc-back .

docker-run: ## Run Docker container
	docker run -it --rm --gpus all -v $(PWD):/workspace arc-back

# Documentation
docs: ## Generate documentation (if using sphinx)
	@echo "Documentation generation not yet implemented"
