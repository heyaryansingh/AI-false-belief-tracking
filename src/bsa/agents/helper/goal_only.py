"""Goal-only helper agent implementation."""

from typing import Any, Dict, List, Optional

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
        """Detect false belief (always False for goal-only helper).

        Goal-only helper assumes human beliefs match true state, so cannot
        detect false beliefs.

        Args:
            observation: Current observation (unused)
            episode_step: Optional episode step (unused)

        Returns:
            False (cannot detect false beliefs)
        """
        return False
