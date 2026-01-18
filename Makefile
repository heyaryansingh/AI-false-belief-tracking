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

test-unit:
	pytest tests/ -k "not integration" -v

test-integration:
	pytest tests/test_integration.py -v

test-coverage:
	pytest tests/ --cov=src/bsa --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -k "not integration and not virtualhome" -v

test-all:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=src/bsa --cov-report=term-missing

coverage-html:
	pytest tests/ --cov=src/bsa --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

coverage-report:
	pytest tests/ --cov=src/bsa --cov-report=term-missing | grep -E "(TOTAL|Name)"

lint:
	ruff check src/ tests/

type-check:
	mypy src/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

ci-test:
	pytest tests/ -v --tb=short

ci-lint:
	ruff check src/ tests/
	mypy src/ || true

ci-reproduce:
	bsa reproduce --small || echo "Reproduction failed"

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
