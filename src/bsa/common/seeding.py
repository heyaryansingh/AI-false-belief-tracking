"""Deterministic seeding utilities."""

import random
import numpy as np
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Add other libraries as needed (torch, tf, etc.)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random number generator with optional seed."""
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()
