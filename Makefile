.PHONY: setup ingest build-features train evaluate serve test clean help

# Default Python and Poetry commands
PYTHON := python
POETRY := poetry
PYTEST := pytest

# Configuration paths
CONFIG_DIR := configs
DATA_CONFIG := $(CONFIG_DIR)/data.yaml
FEATURES_CONFIG := $(CONFIG_DIR)/features.yaml
MODEL_CONFIG := $(CONFIG_DIR)/model.yaml

help: ## Display this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup: ## Install dependencies with Poetry
	$(POETRY) install
	$(POETRY) run pre-commit install

ingest: ## Ingest and normalize raw data (placeholder)
	@echo "Ingesting raw data..."
	$(POETRY) run python -m src.ingest.loader --config $(DATA_CONFIG)

clean-data: ## Clean and canonicalize data
	@echo "Cleaning data..."
	$(POETRY) run python -m src.clean.processor --config $(DATA_CONFIG)

build-features: ## Build feature matrix
	@echo "Building features..."
	$(POETRY) run python -m src.features.pipeline --config $(FEATURES_CONFIG)

train: ## Train model with cross-validation
	@echo "Training model..."
	$(POETRY) run python -m src.model.trainer --config $(MODEL_CONFIG)

evaluate: ## Evaluate model and generate model card
	@echo "Evaluating model..."
	$(POETRY) run python -m src.eval.evaluator --config $(MODEL_CONFIG)

serve: ## Start FastAPI server
	@echo "Starting server..."
	$(POETRY) run uvicorn src.app.server:app --reload --host 0.0.0.0 --port 8000

test: ## Run unit tests
	$(POETRY) run $(PYTEST)

test-verbose: ## Run unit tests with verbose output
	$(POETRY) run $(PYTEST) -v -s

test-coverage: ## Run tests with coverage report
	$(POETRY) run $(PYTEST) --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting checks
	$(POETRY) run black --check src tests
	$(POETRY) run isort --check-only src tests
	$(POETRY) run flake8 src tests

format: ## Format code with black and isort
	$(POETRY) run black src tests
	$(POETRY) run isort src tests

type-check: ## Run mypy type checking
	$(POETRY) run mypy src

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf data/processed/* data/features/*
	rm -rf mlruns/*
	rm -rf models/*

docker-build: ## Build Docker image
	docker build -t tennis-predictor .

docker-run: ## Run Docker container
	docker run -p 8000:8000 tennis-predictor

# Development targets
dev-setup: setup ## Set up development environment
	$(POETRY) run pre-commit install

ci: lint type-check test ## Run all CI checks

# Pipeline targets
pipeline: ingest clean-data build-features train evaluate ## Run full training pipeline

quick-test: ## Run a subset of fast tests
	$(POETRY) run $(PYTEST) -m "not slow"