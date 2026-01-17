"""Base helper agent interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ...common.types import Action, EpisodeStep, Observation


class HelperAgent(ABC):
    """Abstract base class for helper agents.

    Helper agents observe the environment and assist the human agent by:
    - Planning actions based on observations
    - Optionally tracking human's goal and belief state
    - Detecting false beliefs and intervening appropriately

    Different implementations:
    - ReactiveHelper: Reacts to visible objects without inference
    - GoalOnlyHelper: Infers goal but assumes beliefs match true state
    - BeliefSensitiveHelper: Tracks both goal and belief state using particle filter
    """

    @abstractmethod
    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on observation and optional episode context.

        Args:
            observation: Current observation from environment (helper's view)
            episode_step: Optional episode step containing true object locations
                and human beliefs (for debugging/evaluation). Helpers should
                primarily use observations, not privileged information.

        Returns:
            Next action to take
        """
        pass

    @abstractmethod
    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update internal belief state based on human action and observation.

        For reactive helpers, this is a no-op. For goal-only helpers, this
        updates goal inference. For belief-sensitive helpers, this updates
        the particle filter.

        Args:
            observation: Current observation from environment
            human_action: Action taken by human agent
            episode_step: Optional episode step with additional context
        """
        pass

    @abstractmethod
    def get_belief_state(self) -> Optional[Dict[str, Any]]:
        """Return current belief state.

        Returns:
            None for reactive helpers (no belief tracking)
            Dict with belief information for goal-only and belief-sensitive helpers
            (e.g., {"goal_distribution": {...}} or {"goal_distribution": {...}, "object_locations": {...}})
        """
        pass

    def reset(self) -> None:
        """Reset helper state to initial conditions.

        Default implementation: no-op. Override in subclasses that maintain state.
        """
        pass

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> bool:
        """Detect if human has a false belief about object locations.

        Default implementation: returns False (cannot detect false beliefs).
        Override in belief-sensitive helpers.

        Args:
            observation: Current observation
            episode_step: Optional episode step with true locations and human beliefs

        Returns:
            True if false belief detected, False otherwise
        """
        return False
