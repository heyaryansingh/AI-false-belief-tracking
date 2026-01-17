"""Experiment runner."""

from pathlib import Path
from typing import Dict, Any, Optional


def generate_episodes(config: Dict[str, Any], output_dir: Optional[Path] = None, seed: Optional[int] = None) -> None:
    """Generate episodes."""
    # TODO: Implement episode generation
    print(f"Generating episodes with config: {config}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)


def run_experiments(config: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    """Run experiments."""
    # TODO: Implement experiment runner
    print(f"Running experiments with config: {config}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)


def reproduce(small: bool = False) -> None:
    """Full reproduction pipeline."""
    # TODO: Implement reproduction
    print(f"Running reproduction (small={small})")
