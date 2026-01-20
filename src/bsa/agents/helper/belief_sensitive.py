"""Belief-sensitive helper agent implementation.

FIXED VERSION: Uses actual belief inference for interventions.
- High intervention rate when false belief detected
- Uses particle filter to track human's beliefs
- Proactively corrects false beliefs
"""

from typing import Any, Dict, List, Optional

from ...common.types import Action, EpisodeStep, Observation, Task
from ...envs.gridhouse.tasks import get_task, list_tasks
from ...inference.belief import BeliefInference
from .base import HelperAgent
from .policies import InterventionPolicy


class BeliefSensitiveHelper(HelperAgent):
    """Belief-sensitive helper that tracks human's goal and belief state.

    This is the main contribution helper. It:
    - Tracks human's goal using particle filter
    - Tracks human's beliefs about object locations using particle filter
    - Detects false beliefs (believed location != true location)
    - Intervenes proactively when false beliefs detected
    - Assists with goal when no false beliefs

    FIXED VERSION: Now uses proper belief inference:
    - Particle filter initialized with priors (not true locations)
    - Intervention based on inferred belief divergence
    - High intervention rate when false belief detected (60-80%)
    - Lower rate otherwise (20%)
    """

    def __init__(
        self,
        task_list: Optional[List[Task]] = None,
        num_particles: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize belief-sensitive helper.

        Args:
            task_list: List of possible tasks (if None, uses all tasks from tasks.py)
            num_particles: Number of particles for particle filter
            seed: Random seed for reproducibility
        """
        if task_list is None:
            task_list = [get_task(task_id) for task_id in list_tasks()]

        self.belief_inference = BeliefInference(task_list, num_particles, seed)
        self.intervention_policy = InterventionPolicy()
        self.task_list = task_list
        self._last_false_belief_confidence = 0.0
        self._step_count = 0

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on belief state and intervention policy.

        Belief-sensitive policy:
        1. Update belief inference from human action
        2. Compute false belief confidence
        3. If high confidence in false belief -> high probability of intervention
        4. If low confidence -> lower probability, focus on goal assistance

        Args:
            observation: Current observation from environment
            episode_step: Optional episode step (used to update beliefs)

        Returns:
            Next action to take
        """
        self._step_count += 1

        # Update belief inference if episode_step available
        if episode_step is not None:
            self.update_belief(
                observation, episode_step.human_action, episode_step
            )

        # Get false belief confidence
        fb_confidence = self.compute_false_belief_confidence(episode_step)
        self._last_false_belief_confidence = fb_confidence

        # Intervention probability based on false belief confidence
        # High confidence in false belief -> high intervention rate
        # Low confidence -> still intervene sometimes for goal assistance
        if fb_confidence > 0.7:
            intervention_prob = 0.8  # 80% if confident about false belief
        elif fb_confidence > 0.5:
            intervention_prob = 0.5  # 50% if moderate confidence
        else:
            intervention_prob = 0.15  # 15% baseline for goal assistance

        # Get most likely goal for action selection
        most_likely_goal_id = self.belief_inference.get_most_likely_goal()
        if most_likely_goal_id is None:
            return Action.WAIT

        goal = get_task(most_likely_goal_id)

        # Decide whether to intervene
        import numpy as np
        rng = np.random.default_rng()

        if rng.random() > intervention_prob:
            return Action.WAIT

        # Intervene: choose action based on belief state
        if fb_confidence > 0.5:
            # False belief detected - try to communicate or move objects
            # Priority: SAY (communicate), PICKUP (relocate), MOVE (guide)
            return Action.SAY  # Communicate the true location

        # No false belief - help with goal
        for obj_id in goal.critical_objects:
            if obj_id in observation.visible_objects:
                return Action.PICKUP

        if observation.visible_containers:
            return Action.OPEN

        return Action.MOVE

    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update belief inference from human action.

        Args:
            observation: Current observation
            human_action: Action taken by human agent
            episode_step: Optional episode step with true locations and context
        """
        # Get true locations from episode_step (for visibility updates only)
        true_locations = {}
        if episode_step is not None:
            true_locations = episode_step.true_object_locations

        # Update belief inference
        self.belief_inference.update(
            human_action, observation, true_locations, episode_step
        )

    def get_belief_state(self) -> Dict[str, Any]:
        """Return full belief state."""
        return self.belief_inference.get_belief_state()

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
        threshold: float = 0.5,
    ) -> bool:
        """Detect if human has false belief about object locations."""
        return self.compute_false_belief_confidence(episode_step) >= threshold

    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute confidence score for false belief detection.

        Uses the particle filter's INFERRED beliefs about what the human believes,
        compared against the true object locations.

        FIXED: This now uses proper inference without data leakage.
        The particle filter is initialized with priors and only updated
        based on visibility observations.

        Args:
            episode_step: Optional episode step with true locations (for comparison)

        Returns:
            Confidence score in [0, 1] based on inferred beliefs
        """
        if episode_step is None:
            return self._last_false_belief_confidence

        true_locations = episode_step.true_object_locations

        # Use belief inference to compute confidence
        # This compares the particle filter's distribution over locations
        # against the true locations
        return self.belief_inference.compute_false_belief_confidence(
            true_locations, human_believed_locations=None
        )

    def reset(self) -> None:
        """Reset belief inference to initial state."""
        self.belief_inference.reset()
        self._last_false_belief_confidence = 0.0
        self._step_count = 0
