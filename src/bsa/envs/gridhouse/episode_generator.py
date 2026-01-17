"""Episode generator for GridHouse."""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from ...common.types import Episode, EpisodeStep, Goal, Action
from ...common.seeding import get_rng
from .env import GridHouseEnvironment
from .tasks import get_task, list_tasks


class GridHouseEpisodeGenerator:
    """Generate episodes with false-belief interventions."""

    def __init__(
        self,
        env: GridHouseEnvironment,
        seed: Optional[int] = None,
        tau_range: tuple[int, int] = (5, 20),
        occlusion_severity: float = 0.5,
        drift_probability: float = 0.5,
    ):
        """Initialize episode generator.

        Args:
            env: GridHouse environment instance
            seed: Random seed
            tau_range: Range of intervention timesteps
            occlusion_severity: Severity of occlusion (0-1)
            drift_probability: Probability of false-belief intervention
        """
        self.env = env
        self.rng = get_rng(seed)
        self.tau_range = tau_range
        self.occlusion_severity = occlusion_severity
        self.drift_probability = drift_probability

    def generate_episode(
        self,
        goal_id: Optional[str] = None,
        tau: Optional[int] = None,
        intervention_type: Optional[str] = None,
    ) -> Episode:
        """Generate a single episode.

        Args:
            goal_id: Task goal ID (if None, randomly selected)
            tau: Intervention timestep (if None, randomly selected)
            intervention_type: Type of intervention (if None, randomly selected)

        Returns:
            Generated episode
        """
        # Reset environment
        self.env.reset(seed=self.rng.integers(0, 2**31))

        # Select goal
        if goal_id is None:
            goal_id = self.rng.choice(list_tasks())
        task = get_task(goal_id)

        # Select intervention parameters
        if tau is None:
            tau = self.rng.integers(*self.tau_range)
        if intervention_type is None:
            intervention_type = "relocate" if self.rng.random() < self.drift_probability else None

        # Generate episode steps
        steps: List[EpisodeStep] = []
        max_steps = 50

        # Track human belief state (simplified)
        human_belief_locations = self.env.get_object_locations().copy()

        for t in range(max_steps):
            # Apply intervention at tau
            if t == tau and intervention_type == "relocate":
                self._apply_intervention(task, human_belief_locations)

            # Human action (simplified scripted policy)
            human_action = self._get_human_action(task, human_belief_locations, t)

            # Execute action
            human_obs, _, done, _ = self.env.step(human_action, "human")

            # Update human belief (only if they observe)
            self._update_human_belief(human_belief_locations, human_obs)

            # Helper observation
            helper_obs = self.env.get_visible_state("helper")

            # Create step
            step = EpisodeStep(
                episode_id=f"ep_{len(steps)}",
                timestep=t,
                human_action=human_action,
                helper_obs=helper_obs,
                human_obs=human_obs,
                visible_objects_h=human_obs.visible_objects,
                visible_objects_helper=helper_obs.visible_objects,
                true_object_locations=self.env.get_object_locations(),
                human_belief_object_locations=human_belief_locations,
                goal_id=goal_id,
                tau=tau if t >= tau else None,
                intervention_type=intervention_type if t >= tau else None,
            )
            steps.append(step)

            if done:
                break

        return Episode(
            episode_id=f"episode_{self.rng.integers(0, 1000000)}",
            goal_id=goal_id,
            tau=tau,
            intervention_type=intervention_type,
            steps=steps,
            metadata={"max_steps": max_steps, "occlusion_severity": self.occlusion_severity},
        )

    def _apply_intervention(self, task, human_belief_locations: Dict) -> None:
        """Apply false-belief intervention."""
        # Move a critical object while human cannot see
        # Simplified: move object to different location
        for obj_id in task.critical_objects:
            if obj_id in self.env.object_locations:
                # Move object to different location
                # TODO: Implement proper relocation logic
                pass

    def _get_human_action(self, task, human_belief_locations: Dict, t: int) -> Action:
        """Get human action based on task and belief."""
        # Simplified scripted policy
        # TODO: Implement proper planning based on belief state
        if t % 3 == 0:
            return Action.MOVE
        elif t % 3 == 1:
            return Action.OPEN
        else:
            return Action.WAIT

    def _update_human_belief(self, human_belief_locations: Dict, obs) -> None:
        """Update human belief based on observation."""
        # Update belief only if object is observed
        # TODO: Implement proper belief update logic
        pass
