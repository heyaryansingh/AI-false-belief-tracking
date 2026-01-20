"""Reactive helper agent implementation.

FIXED VERSION: Distinct behavior from other helpers.
- Purely reactive with no inference
- Low intervention rate (random)
- No belief tracking
"""

from typing import Any, Dict, Optional
import numpy as np

from ...common.types import Action, EpisodeStep, Observation
from .base import HelperAgent


class ReactiveHelper(HelperAgent):
    """Reactive helper that reacts randomly without goal or belief inference.

    This is the simplest baseline helper. It:
    - Does NOT infer the human's goal
    - Does NOT track the human's beliefs
    - Intervenes randomly with low probability
    - Provides a lower bound on performance

    FIXED: Now has distinct behavior from other helpers:
    - Low intervention rate (20% per step)
    - Random action selection when intervening
    - No goal-directed behavior
    """

    def __init__(self, seed: Optional[int] = None, intervention_rate: float = 0.2):
        """Initialize reactive helper.

        Args:
            seed: Random seed for reproducibility
            intervention_rate: Probability of intervening each step (default 0.2)
        """
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.intervention_rate = intervention_rate

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on random intervention.

        Reactive policy:
        - With low probability, take a random action
        - Otherwise, WAIT (don't interfere)

        This models a helper that doesn't understand the human's goals
        and just occasionally tries to help randomly.

        Args:
            observation: Current observation from environment (helper's view)
            episode_step: Optional episode step (not used)

        Returns:
            Next action to take
        """
        self._step_count += 1

        # With low probability, intervene randomly
        if self.rng.random() < self.intervention_rate:
            # Random action from available actions
            possible_actions = [Action.MOVE, Action.PICKUP, Action.OPEN]

            # Bias toward visible objects/containers if available
            if observation.visible_objects:
                possible_actions = [Action.PICKUP, Action.MOVE]
            elif observation.visible_containers:
                possible_actions = [Action.OPEN, Action.MOVE]

            return self.rng.choice(possible_actions)

        # Default: don't interfere
        return Action.WAIT

    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update belief state (no-op for reactive helper)."""
        pass

    def get_belief_state(self) -> Optional[dict]:
        """Return belief state (None for reactive helper)."""
        return None

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> bool:
        """Detect false belief (random guess for reactive helper)."""
        return self.compute_false_belief_confidence(episode_step) >= 0.5

    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute false belief confidence (random baseline).

        Reactive helper has no inference capability.
        Returns random values centered around 0.5.

        Args:
            episode_step: Optional episode step (mostly unused)

        Returns:
            Random probability score in [0.2, 0.8]
        """
        # Pure random baseline with some variance
        base = 0.5
        noise = self.rng.normal(0, 0.2)

        # Small adjustment based on timestep (later = slightly more likely)
        time_adjustment = 0.0
        if episode_step is not None and episode_step.timestep > 10:
            time_adjustment = 0.05

        confidence = base + noise + time_adjustment
        return max(0.1, min(0.9, confidence))

    def reset(self) -> None:
        """Reset helper state."""
        self._step_count = 0
