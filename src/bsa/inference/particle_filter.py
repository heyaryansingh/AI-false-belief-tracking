"""Particle filter for online belief tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..common.types import Action, EpisodeStep, ObjectLocation, Observation, Task
from ..common.seeding import get_rng
from .likelihood import LikelihoodModel


@dataclass
class Particle:
    """Particle representing a hypothesis about goal and object locations."""

    goal_id: str
    object_locations: Dict[str, ObjectLocation] = field(default_factory=dict)
    weight: float = 1.0


class ParticleFilter:
    """Particle filter for online belief tracking over (goal, object_locations).

    Maintains a distribution over possible goals and object locations by:
    1. Initializing particles uniformly over goals
    2. Updating particle weights based on observed actions using likelihood model
    3. Resampling particles when effective sample size is low
    4. Extracting belief distributions by marginalizing over particles

    This enables tracking both the human's goal AND their beliefs about object
    locations, which is necessary for detecting false beliefs.
    """

    def __init__(
        self,
        task_list: List[Task],
        num_particles: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize particle filter.

        Args:
            task_list: List of possible tasks/goals
            num_particles: Number of particles to maintain
            seed: Random seed for reproducibility
        """
        self.task_list = task_list
        self.num_particles = num_particles
        self.rng = get_rng(seed)

        # Initialize particles uniformly over goals
        self.particles: List[Particle] = []
        particles_per_goal = num_particles // len(task_list) if task_list else 0
        remainder = num_particles % len(task_list) if task_list else num_particles

        for i, task in enumerate(task_list):
            count = particles_per_goal + (1 if i < remainder else 0)
            for _ in range(count):
                self.particles.append(
                    Particle(
                        goal_id=task.task_id,
                        object_locations={},
                        weight=1.0 / num_particles if num_particles > 0 else 0.0,
                    )
                )

        # Normalize initial weights
        self._normalize_weights()

    def update(
        self,
        human_action: Action,
        observation: Observation,
        true_locations: Dict[str, ObjectLocation],
        likelihood_model: "LikelihoodModel",
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update particle weights based on observed human action.

        For each particle, compute likelihood P(action | goal, believed_locations)
        and update weight accordingly.

        Args:
            human_action: Action taken by human agent
            observation: Current observation
            true_locations: True object locations (for initializing beliefs if needed)
            likelihood_model: Likelihood model for computing P(action | goal, believed_locations)
            episode_step: Optional episode step with additional context
        """
        # Update particle weights using likelihood model
        for particle in self.particles:
            goal = next(t for t in self.task_list if t.task_id == particle.goal_id)

            # Initialize object locations if not set (use true locations as initial belief)
            if not particle.object_locations:
                particle.object_locations = true_locations.copy()

            # Compute likelihood P(action | goal, believed_locations)
            likelihood = likelihood_model.compute(
                human_action, goal, particle.object_locations, observation
            )

            # Update weight: P(goal, locations | action) ∝ P(action | goal, locations) * P(goal, locations)
            particle.weight *= likelihood

        # Normalize weights
        self._normalize_weights()

        # Update object location beliefs based on observation
        # If human sees an object, update beliefs in all particles
        for obj_id in observation.visible_objects:
            if obj_id in true_locations:
                # Human sees object - update belief to match true location
                for particle in self.particles:
                    particle.object_locations[obj_id] = true_locations[obj_id]

        # Resample if effective sample size is low
        if self._effective_sample_size() < self.num_particles / 2:
            self.resample()

    def resample(self) -> None:
        """Resample particles using systematic resampling.

        Particles with higher weights are more likely to be selected.
        This prevents particle degeneracy.
        """
        if not self.particles:
            return

        # Compute cumulative weights
        weights = np.array([p.weight for p in self.particles])
        cumulative = np.cumsum(weights)

        # Systematic resampling
        new_particles: List[Particle] = []
        step = 1.0 / self.num_particles
        u = self.rng.random() * step

        for i in range(self.num_particles):
            # Find particle index using cumulative weights
            idx = np.searchsorted(cumulative, u, side="right")
            idx = min(idx, len(self.particles) - 1)  # Ensure valid index

            # Create new particle with same hypothesis but reset weight
            old_particle = self.particles[idx]
            new_particles.append(
                Particle(
                    goal_id=old_particle.goal_id,
                    object_locations=old_particle.object_locations.copy(),
                    weight=1.0 / self.num_particles,
                )
            )
            u += step

        self.particles = new_particles
        self._normalize_weights()

    def predict(self, observation: Observation) -> None:
        """Predict step: add noise/drift to object locations (optional).

        This models uncertainty about object locations when they're not observed.
        For now, this is a no-op, but can be extended to add noise.

        Args:
            observation: Current observation
        """
        # Optional: add noise to object locations in particles
        # For now, keep locations unchanged (deterministic belief updates)
        pass

    def get_belief_distribution(self) -> Dict[str, float]:
        """Get goal distribution by marginalizing over object locations.

        Returns:
            Dictionary mapping goal_id to probability
        """
        goal_counts: Dict[str, float] = {}
        for particle in self.particles:
            goal_counts[particle.goal_id] = (
                goal_counts.get(particle.goal_id, 0.0) + particle.weight
            )

        # Normalize
        total = sum(goal_counts.values())
        if total > 0:
            return {goal_id: count / total for goal_id, count in goal_counts.items()}
        else:
            # Uniform distribution if all weights zero
            num_goals = len(self.task_list)
            return {
                task.task_id: 1.0 / num_goals if num_goals > 0 else 0.0
                for task in self.task_list
            }

    def get_object_location_beliefs(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """Get object location beliefs by marginalizing over goals.

        Returns:
            Dictionary mapping object_id to {location_id: probability}
            Note: Simplified to room_id as location identifier
        """
        # Group particles by object location hypotheses
        object_beliefs: Dict[str, Dict[str, float]] = {}

        for particle in self.particles:
            for obj_id, obj_loc in particle.object_locations.items():
                if obj_id not in object_beliefs:
                    object_beliefs[obj_id] = {}

                # Use room_id as location identifier
                location_id = obj_loc.room_id
                object_beliefs[obj_id][location_id] = (
                    object_beliefs[obj_id].get(location_id, 0.0) + particle.weight
                )

        # Normalize
        for obj_id in object_beliefs:
            total = sum(object_beliefs[obj_id].values())
            if total > 0:
                object_beliefs[obj_id] = {
                    loc_id: prob / total
                    for loc_id, prob in object_beliefs[obj_id].items()
                }

        return object_beliefs

    def get_most_likely_goal(self) -> Optional[str]:
        """Get most likely goal (MAP estimate).

        Returns:
            Goal ID with highest probability, or None if no goals
        """
        distribution = self.get_belief_distribution()
        if not distribution:
            return None

        return max(distribution.items(), key=lambda x: x[1])[0]

    def get_most_likely_locations(self) -> Dict[str, ObjectLocation]:
        """Get most likely object locations (MAP estimate).

        Returns:
            Dictionary mapping object_id to most likely ObjectLocation
        """
        # Find particles with highest weight for each object
        most_likely: Dict[str, ObjectLocation] = {}

        for particle in self.particles:
            for obj_id, obj_loc in particle.object_locations.items():
                if obj_id not in most_likely:
                    most_likely[obj_id] = obj_loc
                else:
                    # Compare weights - keep location from particle with higher weight
                    # Simplified: use first occurrence (could be improved with proper MAP)
                    pass

        return most_likely

    def reset(self) -> None:
        """Reset particle filter to initial uniform distribution."""
        # Reinitialize particles uniformly
        self.__init__(self.task_list, self.num_particles, seed=None)

    def _normalize_weights(self) -> None:
        """Normalize particle weights to sum to 1."""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for particle in self.particles:
                particle.weight /= total
        else:
            # All weights zero - reset to uniform
            uniform_weight = 1.0 / len(self.particles) if self.particles else 0.0
            for particle in self.particles:
                particle.weight = uniform_weight

    def _effective_sample_size(self) -> float:
        """Compute effective sample size for resampling decision.

        Returns:
            Effective sample size (lower = more degeneracy)
        """
        if not self.particles:
            return 0.0

        weights = np.array([p.weight for p in self.particles])
        weights_squared = weights**2
        ess = 1.0 / np.sum(weights_squared) if np.sum(weights_squared) > 0 else 0.0
        return ess
