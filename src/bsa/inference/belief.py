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
        self, true_locations: Dict[str, ObjectLocation], threshold: float = 0.5
    ) -> bool:
        """Detect if human has false belief about object locations.

        Uses probability-based detection with configurable threshold.

        Args:
            true_locations: True object locations
            threshold: Detection threshold (0-1), higher = more conservative

        Returns:
            True if false belief detected with confidence above threshold
        """
        confidence = self.compute_false_belief_confidence(true_locations)
        return confidence >= threshold

    def compute_false_belief_confidence(
        self,
        true_locations: Dict[str, ObjectLocation],
        human_believed_locations: Optional[Dict[str, ObjectLocation]] = None,
    ) -> float:
        """Compute confidence score for false belief detection.

        Returns probability that human has false belief, based on comparing
        the human's believed locations with the true locations.

        Args:
            true_locations: True object locations
            human_believed_locations: Human's believed object locations (if available)

        Returns:
            Confidence score in [0, 1] indicating probability of false belief
        """
        # If we have direct access to human's beliefs, use them for ground truth comparison
        if human_believed_locations:
            false_belief_probs = []

            for task in self.task_list:
                for obj_id in task.critical_objects:
                    if obj_id not in true_locations:
                        continue
                    if obj_id not in human_believed_locations:
                        continue

                    true_loc = true_locations[obj_id]
                    believed_loc = human_believed_locations[obj_id]

                    # Check if belief diverges from truth
                    if true_loc.room_id != believed_loc.room_id:
                        # Strong signal of false belief - locations differ
                        false_belief_probs.append(0.95)
                    elif true_loc.container_id != believed_loc.container_id:
                        # Moderate signal - same room but different container
                        false_belief_probs.append(0.7)
                    else:
                        # Beliefs match truth
                        false_belief_probs.append(0.05)

            if false_belief_probs:
                return max(false_belief_probs)

        # Fallback: Use particle filter distribution
        object_beliefs = self.particle_filter.get_object_location_beliefs()

        if not object_beliefs:
            return 0.0

        # Compute probability of false belief for each critical object
        false_belief_probs = []

        for task in self.task_list:
            for obj_id in task.critical_objects:
                if obj_id not in true_locations:
                    continue

                true_loc = true_locations[obj_id]
                true_room = true_loc.room_id

                # Get probability distribution over locations for this object
                if obj_id not in object_beliefs:
                    continue

                location_probs = object_beliefs[obj_id]

                # Probability that human believes object is NOT in true location
                prob_false_belief = 1.0 - location_probs.get(true_room, 0.0)

                false_belief_probs.append(prob_false_belief)

        # Return maximum probability (most confident false belief)
        if false_belief_probs:
            return max(false_belief_probs)
        return 0.0

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
