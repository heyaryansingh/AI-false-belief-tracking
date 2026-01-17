# Belief-Sensitive Embodied Assistance Research

Research-grade repository implementing a VirtualHome-based benchmark and methods for belief-sensitive embodied assistance under object-centered false belief (Theory of Mind).

## Overview

This repository implements a system where a "human" agent performs long-horizon tasks in a household simulator (VirtualHome) while holding false beliefs about object locations due to occlusion/partial observability. A helper agent observes, infers both the human's goal AND belief state, detects false beliefs, and assists via actions or communication.

**Core Research Question:** Can belief-sensitive assistance (using particle filter/Bayesian inference) outperform reactive and goal-only baselines on false-belief detection, task completion, and wasted action reduction?

## Quick Start

### Prerequisites

- Python 3.9+
- VirtualHome (optional, GridHouse fallback available)
- Make (optional, scripts available)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd belief-assistance-research

# Install package
pip install -e .

# Setup VirtualHome (optional)
bash scripts/setup_virtualhome.sh

# Run minimal reproduction (GridHouse only)
make reproduce
```

### Basic Usage

```bash
# Generate episodes
bsa generate --config configs/generator/default.yaml

# Run experiments
bsa run --config configs/experiments/exp_main.yaml

# Analyze results
bsa analyze --config configs/analysis/plots.yaml

# Full reproduction
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
├── data/                 # Generated episodes
├── results/              # Experiment results
└── tests/                # Test suite
```

## Methodology

### False-Belief Tasks

False-belief tasks are classic Theory of Mind paradigms from cognitive science. In this work, we adapt them to embodied assistance scenarios:

1. **Setup**: Human agent has a goal (e.g., "prepare meal") requiring a task-critical object
2. **Initial State**: Object is at location L0, human observes this
3. **Intervention**: At time τ, object moves to L1 while human cannot see (occlusion/partial observability)
4. **False Belief**: Human continues to believe object is at L0
5. **Helper Task**: Helper agent must detect false belief and assist appropriately

### Belief-Sensitive Assistance

The helper agent maintains a particle filter over:
- **Goal**: Distribution over possible goals (e.g., prepare_meal, set_table)
- **Belief State**: Distribution over object locations as believed by the human

Key capabilities:
- **Belief Tracking**: Online inference of human's belief state from actions
- **False-Belief Detection**: Detect when `argmax(believed_location) != true_location`
- **Intervention Policy**: Choose between fetching object, communicating correction, or opening container

### Baselines

1. **Reactive**: Reacts to visible objects, fetches if needed
2. **Goal-Only**: Infers goal, assumes human beliefs match true state

## Experiments

### Conditions

- **Control**: No relocation (baseline performance)
- **False-Belief**: Relocation unseen by human
- **Seen Relocation**: Relocation seen by human (control for visibility)

### Ablations

- Occlusion severity (visibility radius / room separation)
- τ distribution (early vs late intervention)
- Intervention cost (pragmatics of when to interrupt)
- Particle count (PF compute vs accuracy tradeoff)

## Metrics

- **False-Belief Detection**: AUROC, detection delay (t_detect - τ), false positive rate
- **Belief Tracking**: Location accuracy, cross-entropy, Brier score
- **Task Performance**: Completion steps/time, wasted actions
- **Intervention Quality**: Over/under-correction, precision/recall, timing distribution

## Results

Results are saved to `results/`:
- `results/metrics/*.csv` - Raw metrics
- `results/figures/*.png` - Plots
- `results/tables/*.md` - Tables
- `results/reports/report.md` - Technical report

## Extending the Repository

### Adding New Tasks

1. Define task in `src/bsa/envs/{virtualhome,gridhouse}/tasks.py`
2. Add task-critical objects to config
3. Update episode generator

### Adding New Helper Models

1. Implement `src/bsa/agents/helper/base.py` interface
2. Add config in `configs/models/`
3. Register in experiment config

### Adding New Metrics

1. Implement metric in `src/bsa/metrics/`
2. Add to analysis config
3. Update report template

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_particle_filter.py

# Run with coverage
pytest --cov=src/bsa tests/
```

## CI/CD

GitHub Actions runs:
- Linting (ruff)
- Tests
- Minimal reproduction (`bsa reproduce --small`)

## Citation

```bibtex
@software{belief_assistance_research,
  title = {Belief-Sensitive Embodied Assistance Research},
  author = {Your Name},
  year = {2024},
  url = {<repo-url>}
}
```

## License

[Specify license]

## Contributing

[Contributing guidelines]
