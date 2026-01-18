"""Goal-only helper agent implementation."""

from typing import Any, Dict, List, Optional
import numpy as np

from ...common.types import Action, EpisodeStep, Observation, Task
from ...envs.gridhouse.tasks import get_task, list_tasks
from ..helper.base import HelperAgent
from ...inference.goal import GoalInference


class GoalOnlyHelper(HelperAgent):
    """Goal-only helper that infers human's goal but assumes beliefs match true state.

    This helper:
    - Infers the human's goal from their actions using Bayesian inference
    - Assumes human knows true object locations (no belief tracking)
    - Plans actions to help with inferred goal
    - Cannot detect false beliefs (assumes beliefs = true state)

    This serves as a baseline demonstrating the value of goal inference alone.
    """

    def __init__(self, task_list: Optional[List[Task]] = None, seed: Optional[int] = None):
        """Initialize goal-only helper.

        Args:
            task_list: List of possible tasks (if None, uses all tasks from tasks.py)
            seed: Random seed for goal inference
        """
        if task_list is None:
            from ...envs.gridhouse.tasks import get_task
            task_list = [get_task(task_id) for task_id in list_tasks()]

        self.goal_inference = GoalInference(task_list, seed=seed)
        self.task_list = task_list
        self.rng = np.random.default_rng(seed)
        self._step_count = 0

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on inferred goal.

        Policy:
        - Get most likely goal from inference
        - Plan action to help with inferred goal (assume human beliefs = true state)
        - Fetch critical objects for inferred goal, place them at goal locations

        Args:
            observation: Current observation from environment
            episode_step: Optional episode step (used to update goal inference)

        Returns:
            Next action to take
        """
        # Update goal inference if episode_step available
        if episode_step is not None:
            self.update_belief(observation, episode_step.human_action, episode_step)

        # Get most likely goal
        most_likely_goal_id = self.goal_inference.get_most_likely_goal()
        if most_likely_goal_id is None:
            # No goal inferred yet - explore
            return Action.MOVE

        goal = get_task(most_likely_goal_id)

        # Plan action to help with inferred goal
        # Assume human knows true object locations (no belief tracking)
        # Fetch critical objects for goal

        # If critical objects are visible, try to pick them up
        for obj_id in goal.critical_objects:
            if obj_id in observation.visible_objects:
                return Action.PICKUP

        # If containers with critical objects are visible, open them
        # (Simplified: if containers visible, try to open)
        if observation.visible_containers:
            return Action.OPEN

        # Otherwise, move towards goal (simplified: just move)
        return Action.MOVE

    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update goal inference distribution.

        Args:
            observation: Current observation
            human_action: Action taken by human agent
            episode_step: Optional episode step with additional context
        """
        self.goal_inference.update(human_action, observation, episode_step)

    def get_belief_state(self) -> Dict[str, Any]:
        """Return goal distribution as belief state.

        Returns:
            Dictionary with goal_distribution key
        """
        return {"goal_distribution": self.goal_inference.get_goal_distribution()}

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> bool:
        """Detect false belief (uninformed for goal-only helper).

        Goal-only helper assumes human beliefs match true state, so cannot
        truly detect false beliefs. Returns result based on goal-informed
        but still essentially random probability.

        Args:
            observation: Current observation (unused)
            episode_step: Optional episode step (unused)

        Returns:
            Boolean based on goal-informed probability
        """
        return self.compute_false_belief_confidence(episode_step) >= 0.5

    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute false belief confidence (goal-informed baseline).

        Goal-only helper has goal tracking but no belief tracking. Returns
        a probability that uses goal information but cannot actually detect
        false beliefs about object locations.

        Uses goal inference confidence to modulate the base probability,
        creating more realistic variance than the reactive baseline.

        Args:
            episode_step: Optional episode step with context

        Returns:
            Goal-informed probability score in [0, 1]
        """
        self._step_count += 1

        # Base probability around 0.5
        base_prob = 0.5

        # Add noise for variance
        noise = self.rng.normal(0, 0.12)

        # Use goal inference confidence to modulate
        goal_adjustment = 0.0
        most_likely_goal = self.goal_inference.get_most_likely_goal()
        if most_likely_goal:
            goal_dist = self.goal_inference.get_goal_distribution()
            goal_confidence = goal_dist.get(most_likely_goal, 0.25)
            # Higher goal confidence slightly increases false belief detection
            # (intuition: confident about goal -> more likely to notice discrepancies)
            goal_adjustment = (goal_confidence - 0.25) * 0.2

        # Observable features adjustment
        feature_adjustment = 0.0
        if episode_step is not None:
            # If tau has passed, increase probability slightly
            if episode_step.tau is not None and episode_step.timestep >= episode_step.tau:
                feature_adjustment += 0.08

            # Fewer visible objects = slightly higher probability
            num_visible = len(getattr(episode_step, 'visible_objects_h', []))
            feature_adjustment += 0.01 * max(0, 3 - num_visible)

        # Combine and clamp
        confidence = base_prob + noise + goal_adjustment + feature_adjustment
        return max(0.1, min(0.9, confidence))

    def reset(self) -> None:
        """Reset helper state."""
        self.goal_inference.reset()
        self._step_count = 0
