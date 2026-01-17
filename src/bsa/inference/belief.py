"""Belief inference module wrapping particle filter."""

from typing import Any, Dict, List, Optional

from ..common.types import Action, EpisodeStep, ObjectLocation, Observation, Task
from .likelihood import RuleBasedLikelihoodModel
from .particle_filter import ParticleFilter


class BeliefInference:
    """Belief inference module that wraps particle filter for belief tracking.

    Provides a higher-level interface for tracking human's goal and belief state
    about object locations. Can detect false beliefs by comparing believed vs
    true locations.
    """

    def __init__(
        self,
        task_list: List[Task],
        num_particles: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize belief inference.

        Args:
            task_list: List of possible tasks/goals
            num_particles: Number of particles for particle filter
            seed: Random seed for reproducibility
        """
        self.task_list = task_list
        self.particle_filter = ParticleFilter(task_list, num_particles, seed)
        self.likelihood_model = RuleBasedLikelihoodModel()

    def update(
        self,
        human_action: Action,
        observation: Observation,
        true_locations: Dict[str, ObjectLocation],
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update beliefs from human action.

        Args:
            human_action: Action taken by human agent
            observation: Current observation
            true_locations: True object locations
            episode_step: Optional episode step with additional context
        """
        self.particle_filter.update(
            human_action, observation, true_locations, self.likelihood_model, episode_step
        )

    def detect_false_belief(
        self, true_locations: Dict[str, ObjectLocation]
    ) -> bool:
        """Detect if human has false belief about object locations.

        Compares most likely believed locations with true locations.
        Returns True if any critical object has false belief.

        Args:
            true_locations: True object locations

        Returns:
            True if false belief detected, False otherwise
        """
        # Get most likely believed locations
        believed_locations = self.particle_filter.get_most_likely_locations()

        # Check each critical object across all tasks
        for task in self.task_list:
            for obj_id in task.critical_objects:
                if obj_id in true_locations and obj_id in believed_locations:
                    true_loc = true_locations[obj_id]
                    believed_loc = believed_locations[obj_id]

                    # Compare locations (using room_id as primary identifier)
                    if true_loc.room_id != believed_loc.room_id:
                        return True  # False belief detected

                    # Also check container_id if present
                    if true_loc.container_id != believed_loc.container_id:
                        return True  # False belief detected

        return False  # No false belief detected

    def get_belief_state(self) -> Dict[str, Any]:
        """Return full belief state.

        Returns:
            Dictionary with goal_distribution and object_location_beliefs
        """
        return {
            "goal_distribution": self.particle_filter.get_belief_distribution(),
            "object_location_beliefs": self.particle_filter.get_object_location_beliefs(),
        }

    def get_most_likely_goal(self) -> Optional[str]:
        """Get most likely goal.

        Returns:
            Goal ID with highest probability, or None if no goals
        """
        return self.particle_filter.get_most_likely_goal()

    def get_most_likely_locations(self) -> Dict[str, ObjectLocation]:
        """Get MAP estimate of object locations.

        Returns:
            Dictionary mapping object_id to most likely ObjectLocation
        """
        return self.particle_filter.get_most_likely_locations()

    def reset(self) -> None:
        """Reset beliefs to initial uniform distribution."""
        self.particle_filter.reset()
