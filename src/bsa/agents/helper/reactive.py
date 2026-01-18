"""Reactive helper agent implementation."""

from typing import Any, Dict, Optional
import numpy as np

from ...common.types import Action, EpisodeStep, Observation
from .base import HelperAgent


class ReactiveHelper(HelperAgent):
    """Reactive helper that reacts to visible objects without goal or belief inference.

    This is the simplest baseline helper. It:
    - Reacts to objects that are visible in the current observation
    - Fetches objects when they're visible and nearby
    - Opens containers when objects are inside
    - Does NOT infer the human's goal
    - Does NOT track the human's beliefs
    - Cannot detect false beliefs (returns uninformed probability)

    This serves as a comparison baseline for more sophisticated helpers.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize reactive helper.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on visible objects.

        Reactive policy:
        - If object is visible and nearby, pick it up
        - If object is in visible container, open container
        - Otherwise, move towards visible objects or explore

        Args:
            observation: Current observation from environment (helper's view)
            episode_step: Optional episode step (not used for planning, only debugging)

        Returns:
            Next action to take
        """
        # If we can see objects, try to interact with them
        if observation.visible_objects:
            # Object is visible - try to pick it up
            # (In practice, we'd check distance, but for now assume we can pickup)
            return Action.PICKUP

        # If we can see containers, try to open them
        if observation.visible_containers:
            # Container is visible - try to open it
            return Action.OPEN

        # No visible objects or containers - explore by moving
        return Action.MOVE

    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update belief state (no-op for reactive helper).

        Reactive helper doesn't track beliefs, so this method does nothing.

        Args:
            observation: Current observation (unused)
            human_action: Action taken by human (unused)
            episode_step: Optional episode step (unused)
        """
        pass

    def get_belief_state(self) -> Optional[dict]:
        """Return belief state (None for reactive helper).

        Reactive helper doesn't track beliefs, so always returns None.

        Returns:
            None (no belief state)
        """
        return None

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> bool:
        """Detect false belief (essentially random for reactive helper).

        Reactive helper cannot truly detect false beliefs since it doesn't track beliefs.
        Returns result based on uninformed probability threshold.

        Args:
            observation: Current observation (unused)
            episode_step: Optional episode step (unused)

        Returns:
            Boolean based on random chance (baseline behavior)
        """
        # Use the uninformed probability score with a 0.5 threshold
        return self.compute_false_belief_confidence(episode_step) >= 0.5

    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute false belief confidence (uninformed baseline).

        Reactive helper has no belief tracking, so returns a probability
        that represents uninformed guessing. This creates variance in
        predictions while maintaining an expected value around 0.5.

        The probability varies based on observable features (objects visible,
        containers open, etc.) but cannot actually detect false beliefs.

        Args:
            episode_step: Optional episode step with observation context

        Returns:
            Uninformed probability score in [0, 1] with some variance
        """
        self._step_count += 1

        # Base probability is around 0.5 (random chance)
        base_prob = 0.5

        # Add small variance based on step count and random noise
        # This creates realistic variance in predictions without actual detection
        noise = self.rng.normal(0, 0.15)

        # Use observable features to create more realistic variance
        feature_adjustment = 0.0
        if episode_step is not None:
            # More visible objects = slightly lower false belief probability
            # (heuristic: if human can see objects, less likely to have false belief)
            num_visible = len(getattr(episode_step, 'visible_objects_h', []))
            feature_adjustment = -0.02 * min(num_visible, 5)

            # If intervention has occurred (tau passed), slightly increase probability
            if episode_step.tau is not None and episode_step.timestep >= episode_step.tau:
                feature_adjustment += 0.1

        # Combine components and clamp to [0.1, 0.9]
        confidence = base_prob + noise + feature_adjustment
        return max(0.1, min(0.9, confidence))

    def reset(self) -> None:
        """Reset helper state."""
        self._step_count = 0
