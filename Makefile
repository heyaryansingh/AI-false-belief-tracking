.PHONY: help setup install dev-install generate run analyze reproduce test lint clean

help:
	@echo "Belief-Sensitive Assistance Research - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup          - Initial setup (install package, setup VirtualHome)"
	@echo "  install        - Install package in production mode"
	@echo "  dev-install    - Install package with dev dependencies"
	@echo "  generate       - Generate episodes"
	@echo "  run            - Run experiments"
	@echo "  analyze        - Analyze results and generate report"
	@echo "  reproduce      - Full reproduction (generate + run + analyze)"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  clean          - Clean generated files"

setup: venv
	@echo "Virtual environment setup complete!"
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)"

venv:
	@echo "Setting up virtual environment..."
	@python scripts/setup_venv.py

install:
	pip install -e .

install-venv:
	@if [ -d "venv" ]; then \
		echo "Installing in virtual environment..."; \
		venv/bin/pip install -e . || venv/Scripts/pip install -e .; \
	else \
		echo "Virtual environment not found. Run 'make venv' first."; \
		exit 1; \
	fi

dev-install:
	pip install -e ".[dev]"

generate:
	bsa generate --config configs/generator/default.yaml

run:
	bsa run --config configs/experiments/exp_main.yaml

analyze:
	bsa analyze --config configs/analysis/plots.yaml

reproduce: generate run analyze
	@echo "Reproduction complete. Check results/reports/report.md"

test:
	pytest tests/

lint:
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
