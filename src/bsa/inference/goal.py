"""Goal inference module using Bayesian inference."""

from typing import Dict, List, Optional

import numpy as np

from ..common.types import Action, EpisodeStep, Observation, Task
from ..common.seeding import get_rng


class GoalInference:
    """Bayesian goal inference from human actions.

    Maintains a probability distribution over possible goals and updates it
    based on observed human actions using Bayesian inference:

    P(goal | action) ∝ P(action | goal) * P(goal)

    Uses rule-based likelihood models to compute P(action | goal).
    """

    def __init__(
        self,
        task_list: List[Task],
        prior: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize goal inference.

        Args:
            task_list: List of possible tasks/goals
            prior: Optional prior distribution over goals (if None, uniform prior)
            seed: Random seed for reproducibility
        """
        self.task_list = task_list
        self.rng = get_rng(seed)

        # Initialize goal distribution
        if prior is None:
            # Uniform prior
            num_tasks = len(task_list)
            self.goal_distribution: Dict[str, float] = {
                task.task_id: 1.0 / num_tasks if num_tasks > 0 else 0.0
                for task in task_list
            }
        else:
            # Custom prior (normalize to ensure probabilities sum to 1)
            total = sum(prior.values())
            self.goal_distribution = {
                goal_id: prob / total if total > 0 else 0.0
                for goal_id, prob in prior.items()
            }

        # Store initial prior for reset
        self.initial_prior = self.goal_distribution.copy()

    def update(
        self,
        human_action: Action,
        observation: Observation,
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update goal distribution based on human action.

        Uses Bayesian update: P(goal | action) ∝ P(action | goal) * P(goal)

        Args:
            human_action: Action taken by human agent
            observation: Current observation
            episode_step: Optional episode step with additional context
        """
        # Compute likelihoods for each goal
        likelihoods: Dict[str, float] = {}
        for task in self.task_list:
            likelihood = self._compute_action_likelihood(
                human_action, task, observation
            )
            likelihoods[task.task_id] = likelihood

        # Bayesian update: P(goal | action) ∝ P(action | goal) * P(goal)
        unnormalized: Dict[str, float] = {}
        for goal_id in self.goal_distribution:
            prior_prob = self.goal_distribution[goal_id]
            likelihood = likelihoods.get(goal_id, 0.0)
            unnormalized[goal_id] = likelihood * prior_prob

        # Normalize probabilities
        total = sum(unnormalized.values())
        if total > 0:
            self.goal_distribution = {
                goal_id: prob / total for goal_id, prob in unnormalized.items()
            }
        else:
            # All probabilities zero - reset to uniform
            num_tasks = len(self.task_list)
            self.goal_distribution = {
                task.task_id: 1.0 / num_tasks if num_tasks > 0 else 0.0
                for task in self.task_list
            }

    def _compute_action_likelihood(
        self, action: Action, goal: Task, observation: Observation
    ) -> float:
        """Compute P(action | goal, observation) using rule-based heuristic.

        Actions that make progress towards goal have higher likelihood.

        Args:
            action: Human action
            goal: Task goal hypothesis
            observation: Current observation

        Returns:
            Likelihood P(action | goal, observation) in [0, 1]
        """
        # High likelihood actions: making progress towards goal
        if action == Action.PICKUP:
            # Picking up objects - check if critical objects are visible
            for obj_id in goal.critical_objects:
                if obj_id in observation.visible_objects:
                    return 0.9  # High likelihood if picking up critical object

        if action == Action.MOVE:
            # Moving - check if moving towards critical objects
            # Simplified: medium likelihood (could be improved with position tracking)
            return 0.6

        if action == Action.OPEN:
            # Opening containers - check if critical objects might be inside
            # Simplified: medium-high likelihood
            return 0.7

        if action == Action.PLACE:
            # Placing objects - check if at goal location
            # Simplified: high likelihood if goal involves placing
            return 0.8

        # Low likelihood for actions unrelated to goal
        if action == Action.WAIT:
            return 0.3

        if action == Action.CLOSE:
            return 0.4

        if action == Action.SAY:
            return 0.2  # Communication actions less informative about goal

        # Default: medium likelihood
        return 0.5

    def get_goal_distribution(self) -> Dict[str, float]:
        """Get current goal distribution.

        Returns:
            Dictionary mapping goal_id to probability
        """
        return self.goal_distribution.copy()

    def get_most_likely_goal(self) -> Optional[str]:
        """Get most likely goal.

        Returns:
            Goal ID with highest probability, or None if no goals
        """
        if not self.goal_distribution:
            return None

        return max(self.goal_distribution.items(), key=lambda x: x[1])[0]

    def reset(self) -> None:
        """Reset goal distribution to initial prior."""
        self.goal_distribution = self.initial_prior.copy()
