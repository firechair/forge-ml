.PHONY: help install setup test lint format clean docker-up docker-down verify

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install ForgeML in development mode
	pip install -e .

setup:  ## Run setup script (creates venv, installs dependencies)
	@bash scripts/setup.sh || pwsh scripts/setup.ps1

test:  ## Run all tests
	pytest tests/ -v

test-coverage:  ## Run tests with coverage report
	pytest tests/ --cov=cli --cov=templates --cov-report=html --cov-report=term

lint:  ## Run linters (flake8, mypy)
	flake8 cli/ --max-line-length=100 --ignore=E203,W503
	mypy cli/ --ignore-missing-imports

format:  ## Format code with black
	black cli/ tests/ --line-length=100

format-check:  ## Check code formatting without making changes
	black cli/ tests/ --check --line-length=100

verify:  ## Verify installation
	python scripts/verify.py

docker-up:  ## Start MLflow infrastructure
	cd infra && docker-compose up -d

docker-down:  ## Stop MLflow infrastructure
	cd infra && docker-compose down

docker-team-up:  ## Start team infrastructure
	cd infra && docker-compose -f docker-compose-team.yml up -d

docker-team-down:  ## Stop team infrastructure
	cd infra && docker-compose -f docker-compose-team.yml down

clean:  ## Clean build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

clean-all: clean  ## Clean everything including venv
	rm -rf venv/

init-sentiment:  ## Quick: Create sentiment project
	mlfactory init sentiment --name my-sentiment-project

init-timeseries:  ## Quick: Create timeseries project
	mlfactory init timeseries --name my-timeseries-project

example:  ## Run basic example workflow
	@echo "Creating example project..."
	@mlfactory init sentiment --name example-project
	@echo ""
	@echo "Project created! Next steps:"
	@echo "  cd example-project"
	@echo "  pip install -r requirements.txt"
	@echo "  python train.py"

pre-commit-install:  ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install

pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

build:  ## Build Python package
	python -m build

publish-test:  ## Publish to Test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

docs:  ## Open documentation
	@echo "Opening documentation..."
	@open README.md || xdg-open README.md || start README.md

mlflow-ui:  ## Open MLflow UI
	@echo "Starting MLflow UI at http://localhost:5000"
	@mlflow ui

check: format-check lint test  ## Run all checks (format, lint, test)

dev: install docker-up  ## Setup development environment
	@echo "Development environment ready!"

.DEFAULT_GOAL := help
