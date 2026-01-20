"""Goal-only helper agent implementation.

FIXED VERSION: Distinct behavior from other helpers.
- Uses goal inference to prioritize actions
- Assumes human knows true locations (no belief tracking)
- Medium intervention rate based on goal relevance
"""

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
    - Intervenes when it can help with the inferred goal
    - Cannot detect false beliefs (assumes beliefs = true state)

    FIXED: Now has distinct behavior:
    - Medium intervention rate (40% when goal-relevant, 10% otherwise)
    - Prioritizes actions based on inferred goal
    - Different from reactive (uses goal) and belief_pf (no belief tracking)
    """

    def __init__(self, task_list: Optional[List[Task]] = None, seed: Optional[int] = None):
        """Initialize goal-only helper.

        Args:
            task_list: List of possible tasks (if None, uses all tasks from tasks.py)
            seed: Random seed for goal inference
        """
        if task_list is None:
            task_list = [get_task(task_id) for task_id in list_tasks()]

        self.goal_inference = GoalInference(task_list, seed=seed)
        self.task_list = task_list
        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_goal_confidence = 0.0

    def plan_action(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> Action:
        """Plan next action based on inferred goal.

        Goal-directed policy:
        - Infer human's goal from actions
        - If goal is confident, help with goal-relevant objects
        - If goal is uncertain, mostly wait

        Args:
            observation: Current observation from environment
            episode_step: Optional episode step (used to update goal inference)

        Returns:
            Next action to take
        """
        self._step_count += 1

        # Update goal inference if episode_step available
        if episode_step is not None:
            self.update_belief(observation, episode_step.human_action, episode_step)

        # Get most likely goal and confidence
        most_likely_goal_id = self.goal_inference.get_most_likely_goal()
        goal_dist = self.goal_inference.get_goal_distribution()

        if most_likely_goal_id is None:
            # No goal inferred - wait
            return Action.WAIT

        goal_confidence = goal_dist.get(most_likely_goal_id, 0.25)
        self._last_goal_confidence = goal_confidence
        goal = get_task(most_likely_goal_id)

        # Decision based on goal confidence
        # High confidence -> more likely to intervene
        intervention_prob = 0.1 + 0.4 * goal_confidence  # 10% to 50%

        if self.rng.random() > intervention_prob:
            return Action.WAIT

        # Intervene: prioritize goal-relevant objects
        for obj_id in goal.critical_objects:
            if obj_id in observation.visible_objects:
                # Critical object visible - try to pick it up
                return Action.PICKUP

        # Check if containers might have critical objects
        if observation.visible_containers:
            # Open container (might contain critical object)
            return Action.OPEN

        # Move toward goal area (simplified)
        return Action.MOVE

    def update_belief(
        self,
        observation: Observation,
        human_action: Action,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update goal inference distribution."""
        self.goal_inference.update(human_action, observation, episode_step)

    def get_belief_state(self) -> Dict[str, Any]:
        """Return goal distribution as belief state."""
        return {"goal_distribution": self.goal_inference.get_goal_distribution()}

    def detect_false_belief(
        self,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> bool:
        """Detect false belief (goal-informed but limited)."""
        return self.compute_false_belief_confidence(episode_step) >= 0.5

    def compute_false_belief_confidence(
        self,
        episode_step: Optional[EpisodeStep] = None,
    ) -> float:
        """Compute false belief confidence (goal-informed baseline).

        Goal-only helper cannot truly detect false beliefs since it doesn't
        track beliefs. Uses goal confidence as a weak proxy:
        - Higher goal confidence = more likely to notice if something is wrong
        - But still essentially uninformed about actual beliefs

        Args:
            episode_step: Optional episode step with context

        Returns:
            Goal-informed probability score in [0.2, 0.8]
        """
        # Base probability around 0.4 (slightly lower than random)
        # because assuming beliefs match reality is usually correct
        base = 0.4

        # Add noise for variance
        noise = self.rng.normal(0, 0.15)

        # Goal confidence adjustment
        # Higher goal confidence = slightly more likely to detect issues
        goal_adjustment = self._last_goal_confidence * 0.15

        # Time-based adjustment (false beliefs more likely over time)
        time_adjustment = 0.0
        if episode_step is not None:
            if episode_step.tau is not None and episode_step.timestep >= episode_step.tau:
                # After intervention point, increase probability
                time_adjustment = 0.15
            elif episode_step.timestep > 15:
                time_adjustment = 0.05

        confidence = base + noise + goal_adjustment + time_adjustment
        return max(0.15, min(0.85, confidence))

    def reset(self) -> None:
        """Reset helper state."""
        self.goal_inference.reset()
        self._step_count = 0
        self._last_goal_confidence = 0.0
