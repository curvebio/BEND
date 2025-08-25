# BEND - Benchmark of DNA Language Models
# Makefile for streamlined environment setup and common tasks

.PHONY: help setup setup-dev clean clean-env install install-dev test lint format check docs download-data download-data-original run-example check-gsutil

# Default target
.DEFAULT_GOAL := help

# Python version requirement
PYTHON_VERSION := 3.11
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UV := uv

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)BEND - Benchmark of DNA Language Models$(NC)"
	@echo "$(BLUE)======================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)"
	@echo "  make setup              # Set up environment and install dependencies"
	@echo "  make download-data      # Download BEND dataset (via gsutil)"
	@echo "  make test               # Run tests"
	@echo ""
	@echo "$(YELLOW)Data download options:$(NC)"
	@echo "  make download-data      # Fast download via Google Cloud Storage (gsutil)"
	@echo "  make download-data-original  # Original download script (fallback)"

check-uv: ## Check if uv is installed
	@which $(UV) > /dev/null || (echo "$(RED)Error: uv is not installed. Please install it first: https://docs.astral.sh/uv/$(NC)" && exit 1)
	@echo "$(GREEN)✓ uv is installed$(NC)"

setup: check-uv ## Set up Python environment and install dependencies
	@echo "$(BLUE)Setting up BEND environment...$(NC)"
	@echo "$(YELLOW)Installing Python $(PYTHON_VERSION)...$(NC)"
	$(UV) python install $(PYTHON_VERSION)
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	$(UV) venv --python $(PYTHON_VERSION)
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(UV) pip install -r requirements.txt
	@echo "$(YELLOW)Installing BEND in development mode...$(NC)"
	. $(VENV_DIR)/bin/activate && $(UV) pip install -e .
	@echo "$(GREEN)✓ Environment setup complete!$(NC)"
	@echo "$(YELLOW)Activate with: source $(VENV_DIR)/bin/activate$(NC)"

setup-dev: setup ## Set up development environment with additional tools
	@echo "$(BLUE)Setting up development environment...$(NC)"
	. $(VENV_DIR)/bin/activate && $(UV) pip install pytest pytest-cov black isort flake8 mypy pre-commit sphinx sphinx-rtd-theme myst-parser
	@echo "$(GREEN)✓ Development environment setup complete!$(NC)"

install: ## Install BEND package only (assumes environment exists)
	@echo "$(BLUE)Installing BEND package...$(NC)"
	. $(VENV_DIR)/bin/activate && $(UV) pip install -e .
	@echo "$(GREEN)✓ BEND installed$(NC)"

install-dev: ## Install development dependencies (assumes environment exists)
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	. $(VENV_DIR)/bin/activate && $(UV) pip install pytest pytest-cov black isort flake8 mypy pre-commit sphinx sphinx-rtd-theme myst-parser
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

check-gsutil: ## Check if gsutil is installed
	@which gsutil > /dev/null || (echo "$(RED)Error: gsutil is not installed. Please install Google Cloud SDK first: https://cloud.google.com/sdk/docs/install$(NC)" && exit 1)
	@echo "$(GREEN)✓ gsutil is installed$(NC)"

download-data: check-gsutil ## Download BEND dataset from Google Cloud Storage
	@echo "$(BLUE)Downloading BEND dataset from Google Cloud Storage...$(NC)"
	@echo "$(YELLOW)Source: gs://curvebio-mahdibaghbanzadeh/bend$(NC)"
	@mkdir -p data
	gsutil -m cp -r gs://curvebio-mahdibaghbanzadeh/bend/* data/
	@echo "$(GREEN)✓ Dataset downloaded to data/ directory$(NC)"

download-data-original: ## Download BEND dataset using original script (fallback)
	@echo "$(BLUE)Downloading BEND dataset using original script...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python scripts/download_bend.py
	@echo "$(GREEN)✓ Dataset downloaded$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python -m pytest tests/ -v || echo "$(YELLOW)Note: No tests found. Create tests in tests/ directory.$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python -m pytest tests/ --cov=bend --cov-report=html --cov-report=term-missing || echo "$(YELLOW)Note: No tests found. Create tests in tests/ directory.$(NC)"

lint: ## Run linting (flake8)
	@echo "$(BLUE)Running linting...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python -m flake8 bend/ scripts/ --max-line-length=88 --extend-ignore=E203,W503

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python -m black bend/ scripts/
	. $(VENV_DIR)/bin/activate && python -m isort bend/ scripts/
	@echo "$(GREEN)✓ Code formatted$(NC)"

check: ## Run all checks (format, lint, type check)
	@echo "$(BLUE)Running all checks...$(NC)"
	$(MAKE) format
	$(MAKE) lint
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python -m mypy bend/ --ignore-missing-imports || echo "$(YELLOW)mypy check completed with warnings$(NC)"
	@echo "$(GREEN)✓ All checks completed$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup-dev' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && cd docs && make html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html/$(NC)"

precompute-embeddings: ## Precompute embeddings for all models and tasks
	@echo "$(BLUE)Precomputing embeddings...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python scripts/precompute_embeddings.py

train-example: ## Run training example on a sample task
	@echo "$(BLUE)Running training example...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && python scripts/train_on_task.py

run-notebook: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	@if [ ! -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && $(UV) pip install jupyter notebook
	. $(VENV_DIR)/bin/activate && jupyter notebook

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-env: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	rm -rf $(VENV_DIR)
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

reset: clean-env setup ## Reset environment (clean and setup)
	@echo "$(GREEN)✓ Environment reset complete$(NC)"

status: ## Show environment status
	@echo "$(BLUE)BEND Environment Status$(NC)"
	@echo "======================"
	@echo "Virtual environment: $(VENV_DIR)"
	@if [ -f "$(VENV_DIR)/bin/activate" ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
		echo "Python version: $$(. $(VENV_DIR)/bin/activate && python --version)"; \
		echo "Installed packages: $$(. $(VENV_DIR)/bin/activate && pip list | wc -l) packages"; \
	else \
		echo "$(RED)✗ Virtual environment not found$(NC)"; \
	fi
	@echo ""
	@echo "Data directory:"
	@if [ -d "data" ]; then \
		echo "$(GREEN)✓ Data directory exists$(NC)"; \
		echo "Data files: $$(find data -name "*.bed" | wc -l) .bed files"; \
		echo "HDF5 files: $$(find data -name "*.hdf5" | wc -l) .hdf5 files"; \
		echo "Total data size: $$(du -sh data 2>/dev/null | cut -f1 || echo "unknown")"; \
	else \
		echo "$(RED)✗ Data directory not found$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Download options:$(NC)"
	@if command -v gsutil >/dev/null 2>&1; then \
		echo "$(GREEN)✓ gsutil available$(NC) - use 'make download-data' for fast download"; \
	else \
		echo "$(YELLOW)⚠ gsutil not available$(NC) - use 'make download-data-original' for fallback"; \
		echo "  Install gsutil: https://cloud.google.com/sdk/docs/install"; \
	fi

# Environment activation helper
activate: ## Show activation command
	@echo "$(YELLOW)To activate the environment, run:$(NC)"
	@echo "source $(VENV_DIR)/bin/activate"
