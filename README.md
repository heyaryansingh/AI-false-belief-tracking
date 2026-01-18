# Belief-Sensitive Embodied Assistance

[![CI](https://github.com/yourusername/belief-assistance-research/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/belief-assistance-research/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Research-grade repository implementing a VirtualHome-based benchmark and methods for belief-sensitive embodied assistance under object-centered false belief (Theory of Mind).

## Overview

This repository implements a system where a "human" agent performs long-horizon tasks in a household simulator (VirtualHome or GridHouse) while holding false beliefs about object locations due to occlusion/partial observability. A helper agent observes, infers both the human's goal AND belief state, detects false beliefs, and assists via actions or communication.

**Core Research Question:** Can belief-sensitive assistance (using particle filter/Bayesian inference) outperform reactive and goal-only baselines on false-belief detection, task completion, and wasted action reduction?

## Features

- **Dual Simulator Support**: VirtualHome (3D) and GridHouse (symbolic fallback)
- **Belief Tracking**: Online particle filter inference of human's goal and object location beliefs
- **False-Belief Detection**: Automatic detection of when human beliefs diverge from reality
- **Comprehensive Metrics**: AUROC, detection latency, task completion, wasted actions, intervention quality
- **Reproducible Research**: Deterministic seeding, manifest tracking, full pipeline automation
- **Analysis Pipeline**: Automated plots, tables, and technical report generation

## Quick Start

### Prerequisites

- Python 3.9-3.11 (recommended for VirtualHome compatibility)
- VirtualHome (optional, GridHouse fallback available)

### Installation

**Recommended: Use virtual environment**

```bash
# Clone repository
git clone https://github.com/yourusername/belief-assistance-research.git
cd belief-assistance-research

# Setup virtual environment
python scripts/setup_venv.py

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install package
pip install -e ".[dev]"

# Verify installation
python -c "from src.bsa.envs.gridhouse import GridHouseEnvironment; print('✓ GridHouse OK')"
```

### Basic Usage

```bash
# Generate episodes
bsa generate --config configs/generator/default.yaml

# Run experiments
bsa run --config configs/experiments/exp_main.yaml

# Analyze results
bsa analyze --config configs/analysis/plots.yaml

# Full reproduction pipeline
make reproduce
```

## Repository Structure

```
belief-assistance-research/
├── src/bsa/              # Main package
│   ├── envs/            # Environment interfaces (VirtualHome, GridHouse)
│   ├── agents/          # Human and helper agents
│   ├── inference/       # Goal and belief inference
│   ├── experiments/     # Experiment runners
│   ├── metrics/         # Evaluation metrics
│   └── analysis/        # Results analysis and plotting
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── data/                 # Generated episodes (gitignored)
├── results/              # Experiment results (gitignored)
└── tests/                # Test suite
```

## Methodology

### False-Belief Tasks

False-belief tasks are classic Theory of Mind paradigms from cognitive science. In this work, we adapt them to embodied assistance scenarios:

1. **Setup**: Human agent has a goal (e.g., "prepare meal") requiring a task-critical object
2. **Initial State**: Object is at location L₀, human observes this
3. **Intervention**: At time τ, object moves to L₁ while human cannot see (occlusion/partial observability)
4. **False Belief**: Human continues to believe object is at L₀
5. **Helper Task**: Helper agent must detect false belief and assist appropriately

### Belief-Sensitive Assistance

The helper agent maintains a particle filter over:
- **Goal**: Distribution over possible goals (e.g., prepare_meal, set_table)
- **Belief State**: Distribution over object locations as believed by the human

Key capabilities:
- **Belief Tracking**: Online inference of human's belief state from actions
- **False-Belief Detection**: Detect when `argmax(believed_location) ≠ true_location`
- **Intervention Policy**: Choose between fetching object, communicating correction, or opening container

### Baselines

1. **Reactive**: Reacts to visible objects, fetches if needed
2. **Goal-Only**: Infers goal, assumes human beliefs match true state

## Experiments

### Conditions

- **Control**: No relocation (baseline performance)
- **False-Belief**: Relocation unseen by human
- **Seen Relocation**: Relocation seen by human (control for visibility)

### Metrics

- **False-Belief Detection**: AUROC, detection delay (t_detect - τ), false positive rate
- **Belief Tracking**: Location accuracy, cross-entropy, Brier score
- **Task Performance**: Completion steps/time, wasted actions
- **Intervention Quality**: Over/under-correction, precision/recall, timing distribution

## Results

Results are saved to `results/`:
- `results/metrics/*.parquet` - Raw metrics
- `results/figures/*.png` - Plots
- `results/tables/*.md` - Tables
- `results/reports/report.md` - Technical report

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/bsa tests/

# Run specific test suite
pytest tests/test_particle_filter.py -v
```

## CI/CD

GitHub Actions runs:
- Linting (ruff)
- Type checking (mypy)
- Tests (pytest)
- Coverage reporting
- Minimal reproduction (`bsa reproduce --small`)

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{belief_assistance_research,
  title = {Belief-Sensitive Embodied Assistance Research},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/heyaryansingh/AI-false-belief-tracking}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- VirtualHome simulator: [VirtualHome](https://github.com/xavierpuigf/virtualhome)
- Theory of Mind research in cognitive science
