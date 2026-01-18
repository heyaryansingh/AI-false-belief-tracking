# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Belief-Sensitive Embodied Assistance (BSA) - A research repository implementing a VirtualHome/GridHouse-based benchmark for belief-sensitive embodied assistance under object-centered false belief (Theory of Mind).

The core research question: Can belief-sensitive assistance (using particle filter/Bayesian inference) outperform reactive and goal-only baselines on false-belief detection, task completion, and wasted action reduction?

## Commands

### Development Setup
```bash
python scripts/setup_venv.py          # Create virtual environment
pip install -e ".[dev]"               # Install with dev dependencies
pip install -e ".[virtualhome]"       # Install with VirtualHome support
```

### Testing
```bash
pytest tests/                         # Run all tests
pytest tests/ -k "not integration"    # Unit tests only
pytest tests/ -k "not virtualhome"    # Skip VirtualHome tests
pytest tests/test_particle_filter.py -v  # Run specific test file
pytest --cov=src/bsa tests/           # With coverage
```

### Linting and Type Checking
```bash
ruff check src/ tests/                # Lint
ruff format src/ tests/               # Format
mypy src/                             # Type check
```

### CLI Commands
```bash
bsa generate --config configs/generator/default.yaml    # Generate episodes
bsa run --config configs/experiments/exp_main.yaml      # Run experiments
bsa analyze --config configs/analysis/plots.yaml        # Analyze results
bsa reproduce --small                                   # Minimal reproduction (CI)
bsa sweep --config configs/experiments/sweep_particles.yaml  # Parameter sweep
```

## Architecture

### Core Components

**Environment Layer** (`src/bsa/envs/`)
- `base.py`: Abstract `Environment` interface with `reset()`, `step()`, `get_true_state()`, `get_visible_state()`
- `gridhouse/`: Symbolic grid-based fallback environment (always available)
- `virtualhome/`: 3D VirtualHome simulator integration (optional dependency)

**Agent Layer** (`src/bsa/agents/`)
- `human/`: Scripted human agent with policies that follow task plans
- `helper/`: Helper agent implementations:
  - `ReactiveHelper`: No inference, reacts to visible objects
  - `GoalOnlyHelper`: Infers goal, assumes beliefs match true state
  - `BeliefSensitiveHelper`: Full particle filter over goal AND belief state

**Inference Layer** (`src/bsa/inference/`)
- `particle_filter.py`: Core `ParticleFilter` class maintaining distribution over (goal, object_locations)
- `likelihood.py`: `LikelihoodModel` for computing P(action | goal, believed_locations)
- `goal.py` / `belief.py`: Goal and belief inference utilities

**Experiment Pipeline** (`src/bsa/experiments/`)
- `runner.py`: Episode execution and data collection
- `evaluator.py`: Metrics computation
- `sweep.py`: Parameter sweep orchestration
- `manifest.py`: Reproducibility tracking

### Key Data Types (`src/bsa/common/types.py`)
- `Action`: Enum of agent actions (MOVE, OPEN, CLOSE, PICKUP, PLACE, WAIT, SAY)
- `Observation`: Agent's visible state (visible_objects, current_room, position)
- `EpisodeStep`: Single timestep with human/helper observations, true vs believed locations
- `ParticleFilter.Particle`: Hypothesis with goal_id, object_locations, weight

### Configuration
All configs in `configs/`:
- `env/`: Environment settings (gridhouse.yaml)
- `generator/`: Episode generation parameters
- `experiments/`: Experiment configs (exp_main.yaml, sweep_*.yaml)
- `models/`: Agent configs (belief_pf.yaml, goal_only.yaml, reactive.yaml)
- `analysis/`: Plot/table generation settings

## Key Patterns

**Partial Observability**: Human agent has limited visibility (occlusion). Objects can be relocated while unobserved, creating false beliefs.

**Particle Filter Updates**: On each human action, particles are weighted by P(action | goal, believed_locations), normalized, and resampled when ESS drops below threshold.

**False Belief Detection**: Compare `argmax(believed_location)` with `true_location` for task-critical objects.

**Intervention Types**: Helper can fetch object, communicate correction, or open container to reveal object.

## Python Version
Python 3.9-3.11 recommended. 3.12 works for GridHouse but may have VirtualHome compatibility issues.
