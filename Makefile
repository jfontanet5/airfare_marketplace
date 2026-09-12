.DEFAULT_GOAL := help
PY ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create virtualenv and install package with dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

install: $(VENV) ## Alias for venv creation

run: $(VENV) ## Launch the Streamlit app
	$(BIN)/streamlit run airfare/ui/app.py

test: $(VENV) ## Run the test suite with coverage
	$(BIN)/pytest

lint: $(VENV) ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

fmt: $(VENV) ## Auto-format and fix imports
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: $(VENV) ## mypy strict
	$(BIN)/mypy airfare

check: lint typecheck test ## Everything CI runs

train: $(VENV) ## Train the synthetic demo model into models/
	$(BIN)/airfare-train

docker-build: ## Build the container image
	docker build -t airfare-marketplace .

docker-run: ## Run the container on :8501 (offline mode works without a .env)
	docker run --rm -p 8501:8501 --env-file .env airfare-marketplace

clean: ## Remove caches
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install run test lint fmt typecheck check train docker-build docker-run clean
