"""Belief-sensitive helper agent implementation."""

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

    This demonstrates the value of belief-sensitive assistance over reactive
    and goal-only baselines.
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

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on belief state and intervention policy.

        Steps:
        1. Update belief inference from human action (if episode_step available)
        2. Detect false belief using belief inference
        3. Use intervention policy to decide action
        4. Return intervention action or assistive action

        Args:
            observation: Current observation from environment
            episode_step: Optional episode step (used to update beliefs)

        Returns:
            Next action to take
        """
        # Update belief inference if episode_step available
        if episode_step is not None:
            self.update_belief(
                observation, episode_step.human_action, episode_step
            )

        # Get belief state
        belief_state = self.belief_inference.get_belief_state()

        # Get most likely goal
        most_likely_goal_id = self.belief_inference.get_most_likely_goal()
        if most_likely_goal_id is None:
            # No goal inferred yet - explore
            return Action.MOVE

        goal = get_task(most_likely_goal_id)

        # Get true locations from episode_step if available
        true_locations = {}
        if episode_step is not None:
            true_locations = episode_step.true_object_locations

        # Use intervention policy to decide action
        if self.intervention_policy.should_intervene(
            belief_state, true_locations, observation, goal
        ):
            return self.intervention_policy.choose_intervention(
                belief_state, true_locations, observation, goal
            )

        # No intervention needed - wait
        return Action.WAIT

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
        # Get true locations from episode_step
        true_locations = {}
        if episode_step is not None:
            true_locations = episode_step.true_object_locations

        # Update belief inference
        self.belief_inference.update(
            human_action, observation, true_locations, episode_step
        )

    def get_belief_state(self) -> Dict[str, Any]:
        """Return full belief state.

        Returns:
            Dictionary with goal_distribution and object_location_beliefs
        """
        return self.belief_inference.get_belief_state()

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
        threshold: float = 0.5,
    ) -> bool:
        """Detect if human has false belief about object locations.

        Args:
            observation: Current observation (unused but required by interface)
            episode_step: Optional episode step with true locations
            threshold: Detection threshold (0-1)

        Returns:
            True if false belief detected with confidence above threshold
        """
        if episode_step is None:
            return False

        true_locations = episode_step.true_object_locations
        return self.belief_inference.detect_false_belief(true_locations, threshold=threshold)
    
    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute confidence score for false belief detection.

        Uses the particle filter's INFERRED beliefs about what the human believes,
        compared against the true object locations. This is the honest evaluation -
        we do NOT use ground truth about human beliefs (that would be cheating).

        The particle filter must actually infer beliefs from observing human actions.

        Args:
            episode_step: Optional episode step with true locations (for comparison)

        Returns:
            Confidence score in [0, 1] based on inferred beliefs
        """
        if episode_step is None:
            return 0.0

        true_locations = episode_step.true_object_locations

        # IMPORTANT: Do NOT pass human_believed_locations - that would be data leakage!
        # The particle filter must infer beliefs from observations, not use ground truth.
        return self.belief_inference.compute_false_belief_confidence(
            true_locations, human_believed_locations=None
        )

    def reset(self) -> None:
        """Reset belief inference to initial state."""
        self.belief_inference.reset()
