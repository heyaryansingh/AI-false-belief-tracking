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

        Uses particle filter distribution to estimate P(false belief | observations).

        Mathematical formulation:
        Let π_i(r) = P(object i in room r | observations) be the particle distribution.
        The false belief probability for object i is:

            fb_i = 1 - π_i(true_room_i)

        We compute an aggregate score using:
        1. Maximum over critical objects (most likely false belief)
        2. Uncertainty weighting based on particle entropy
        3. Mild smoothing to avoid extreme 0/1 outputs

        The final score has natural variance from:
        - Particle filter stochasticity (resampling, initialization)
        - Object-specific uncertainty differences
        - Episode-specific observation patterns

        Args:
            true_locations: True object locations
            human_believed_locations: Unused (inference-only approach)

        Returns:
            Confidence score in [0.1, 0.9] indicating P(false belief)
        """
        import numpy as np

        # Get particle filter beliefs
        object_beliefs = self.particle_filter.get_object_location_beliefs()

        if not object_beliefs:
            return 0.5

        # Collect false belief evidence for each critical object
        fb_evidence = []
        uncertainties = []

        for task in self.task_list:
            for obj_id in task.critical_objects:
                if obj_id not in true_locations:
                    continue

                true_loc = true_locations[obj_id]
                true_room = true_loc.room_id

                if obj_id not in object_beliefs:
                    fb_evidence.append(0.5)
                    uncertainties.append(1.0)
                    continue

                location_probs = object_beliefs[obj_id]

                # P(false belief) = 1 - P(correct room)
                prob_correct = location_probs.get(true_room, 0.0)
                prob_false = 1.0 - prob_correct

                # Compute entropy-based uncertainty
                probs = np.array(list(location_probs.values()))
                probs = probs[probs > 0]
                if len(probs) > 1:
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    max_entropy = np.log(len(probs))
                    uncertainty = entropy / max_entropy if max_entropy > 0 else 1.0
                else:
                    uncertainty = 0.0

                fb_evidence.append(prob_false)
                uncertainties.append(uncertainty)

        if not fb_evidence:
            return 0.5

        fb_evidence = np.array(fb_evidence)
        uncertainties = np.array(uncertainties)

        # Aggregate: use maximum with uncertainty weighting
        # High uncertainty reduces confidence in the signal
        confidence_weights = 1.0 - 0.5 * uncertainties
        weighted_evidence = fb_evidence * confidence_weights

        # Use maximum (strongest signal) with some contribution from mean
        max_score = np.max(weighted_evidence)
        mean_score = np.mean(weighted_evidence)
        aggregated = 0.7 * max_score + 0.3 * mean_score

        # Light smoothing: map [0,1] to [0.1, 0.9]
        smoothed = 0.1 + 0.8 * aggregated

        return float(smoothed)

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
