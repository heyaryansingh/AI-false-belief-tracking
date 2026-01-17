"""Aggregate and analyze results."""

from pathlib import Path
from typing import Dict, Any


def analyze_results(config: Dict[str, Any], input_dir: Optional[Path] = None) -> None:
    """Analyze experiment results."""
    # TODO: Implement analysis
    print(f"Analyzing results with config: {config}")
    if input_dir:
        print(f"Input directory: {input_dir}")
