# Contributing to Belief-Sensitive Embodied Assistance

Thank you for your interest in contributing! This document provides guidelines for contributing to this research repository.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/belief-assistance-research.git`
3. Create a virtual environment: `python scripts/setup_venv.py`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Run tests: `pytest tests/`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Run linting: `ruff check src/ tests/`
- Format code: `ruff format src/ tests/`

## Testing

- Write tests for all new features
- Ensure all tests pass: `pytest tests/`
- Maintain or improve test coverage
- Add integration tests for new workflows

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass and linting passes
5. Update documentation as needed
6. Submit a pull request with a clear description

## Reporting Issues

Please use GitHub Issues to report bugs or suggest features. Include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## Research Contributions

For research contributions (new models, metrics, experiments):
- Include experimental results
- Update documentation
- Add example configs
- Update technical report template if needed
