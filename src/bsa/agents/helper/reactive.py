"""Reactive helper agent implementation."""

from typing import Optional

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
    - Cannot detect false beliefs

    This serves as a comparison baseline for more sophisticated helpers.
    """

    def __init__(self):
        """Initialize reactive helper (no state to initialize)."""
        pass

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
        """Detect false belief (always False for reactive helper).

        Reactive helper cannot detect false beliefs since it doesn't track beliefs.

        Args:
            observation: Current observation (unused)
            episode_step: Optional episode step (unused)

        Returns:
            False (cannot detect false beliefs)
        """
        return False
