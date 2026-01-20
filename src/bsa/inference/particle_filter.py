"""Particle filter for online belief tracking.

FIXED VERSION: Removes data leakage by:
1. Not initializing with true_locations
2. Using uncertain prior beliefs
3. Adding observation noise
4. Only updating beliefs from inferred observations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common.types import Action, EpisodeStep, ObjectLocation, Observation, Task
from ..common.seeding import get_rng
from .likelihood import LikelihoodModel


# Define possible rooms and containers for prior initialization
ROOMS = ["kitchen", "living_room", "bedroom", "bathroom"]
CONTAINERS = ["cabinet_kitchen", "drawer_kitchen", "table_living", "desk_bedroom"]

# Object priors: which rooms objects are likely to be in
OBJECT_ROOM_PRIORS = {
    "knife": {"kitchen": 0.7, "living_room": 0.1, "bedroom": 0.1, "bathroom": 0.1},
    "plate": {"kitchen": 0.6, "living_room": 0.2, "bedroom": 0.1, "bathroom": 0.1},
    "fork": {"kitchen": 0.7, "living_room": 0.15, "bedroom": 0.1, "bathroom": 0.05},
    "keys": {"kitchen": 0.15, "living_room": 0.25, "bedroom": 0.5, "bathroom": 0.1},
    "book": {"kitchen": 0.1, "living_room": 0.3, "bedroom": 0.5, "bathroom": 0.1},
}


@dataclass
class Particle:
    """Particle representing a hypothesis about goal and object locations."""

    goal_id: str
    object_locations: Dict[str, ObjectLocation] = field(default_factory=dict)
    weight: float = 1.0


class ParticleFilter:
    """Particle filter for online belief tracking over (goal, object_locations).

    FIXED VERSION: This now does real inference without data leakage.

    Key changes:
    1. Particles initialize with PRIOR beliefs, NOT true locations
    2. Beliefs only update when human observes objects (visibility-based)
    3. Observation updates have noise (not 100% certainty)
    4. Different particles can have different location hypotheses

    This enables tracking both the human's goal AND their beliefs about object
    locations, which is necessary for detecting false beliefs.
    """

    def __init__(
        self,
        task_list: List[Task],
        num_particles: int = 100,
        seed: Optional[int] = None,
        observation_noise: float = 0.1,  # Probability of incorrect observation
        prior_noise: float = 0.2,  # How much to vary from prior
    ):
        """Initialize particle filter.

        Args:
            task_list: List of possible tasks/goals
            num_particles: Number of particles to maintain
            seed: Random seed for reproducibility
            observation_noise: Noise in observation updates (0-1)
            prior_noise: Noise in prior initialization (0-1)
        """
        self.task_list = task_list
        self.num_particles = num_particles
        self.rng = get_rng(seed)
        self.observation_noise = observation_noise
        self.prior_noise = prior_noise
        self._seed = seed

        # Track what the human has observed
        self._observed_objects: Dict[str, ObjectLocation] = {}

        # Initialize particles with prior beliefs
        self.particles: List[Particle] = []
        self._initialize_particles()

    def _initialize_particles(self) -> None:
        """Initialize particles with prior beliefs (NOT true locations)."""
        self.particles = []
        particles_per_goal = self.num_particles // len(self.task_list) if self.task_list else 0
        remainder = self.num_particles % len(self.task_list) if self.task_list else self.num_particles

        for i, task in enumerate(self.task_list):
            count = particles_per_goal + (1 if i < remainder else 0)
            for _ in range(count):
                # Sample object locations from priors
                object_locations = self._sample_prior_locations()
                self.particles.append(
                    Particle(
                        goal_id=task.task_id,
                        object_locations=object_locations,
                        weight=1.0 / self.num_particles if self.num_particles > 0 else 0.0,
                    )
                )

        self._normalize_weights()

    def _sample_prior_locations(self) -> Dict[str, ObjectLocation]:
        """Sample object locations from prior distribution.

        Returns locations based on common-sense priors about where objects
        are likely to be, with some noise added.
        """
        locations = {}

        for obj_id, room_probs in OBJECT_ROOM_PRIORS.items():
            # Add noise to priors
            probs = {}
            for room, prob in room_probs.items():
                noise = self.rng.uniform(-self.prior_noise, self.prior_noise)
                probs[room] = max(0.01, prob + noise)  # Keep at least 1% probability

            # Normalize
            total = sum(probs.values())
            probs = {r: p / total for r, p in probs.items()}

            # Sample room
            rooms = list(probs.keys())
            probabilities = [probs[r] for r in rooms]
            room_id = self.rng.choice(rooms, p=probabilities)

            # Randomly assign to container or open floor
            container_id = None
            if self.rng.random() < 0.6:  # 60% chance in container
                # Pick a container in the room or nearby
                possible_containers = [c for c in CONTAINERS if room_id in c]
                if not possible_containers:
                    possible_containers = CONTAINERS
                container_id = self.rng.choice(possible_containers)

            # Random position in room
            room_centers = {
                "kitchen": (5, 5),
                "living_room": (15, 5),
                "bedroom": (5, 15),
                "bathroom": (15, 15),
            }
            center = room_centers.get(room_id, (10, 10))
            x = center[0] + self.rng.integers(-3, 4)
            y = center[1] + self.rng.integers(-3, 4)

            locations[obj_id] = ObjectLocation(
                object_id=obj_id,
                container_id=container_id,
                room_id=room_id,
                position=(float(x), float(y), 0.0),
            )

        return locations

    def update(
        self,
        human_action: Action,
        observation: Observation,
        true_locations: Dict[str, ObjectLocation],
        likelihood_model: "LikelihoodModel",
        episode_step: Optional[EpisodeStep] = None,
    ) -> None:
        """Update particle weights based on observed human action.

        FIXED: No longer uses true_locations for initialization.
        Only uses visibility information to update beliefs when human
        observes objects.

        Args:
            human_action: Action taken by human agent
            observation: Current observation (what human sees)
            true_locations: True object locations (ONLY used for visibility check)
            likelihood_model: Likelihood model for P(action | goal, believed_locations)
            episode_step: Optional episode step (for visibility info)
        """
        # Update particle weights using likelihood model
        for particle in self.particles:
            goal = next(t for t in self.task_list if t.task_id == particle.goal_id)

            # Compute likelihood P(action | goal, believed_locations)
            likelihood = likelihood_model.compute(
                human_action, goal, particle.object_locations, observation
            )

            # Update weight
            particle.weight *= likelihood

        # Normalize weights
        self._normalize_weights()

        # Update beliefs based on what human OBSERVES (with noise)
        if episode_step is not None:
            self._update_beliefs_from_observation(episode_step, true_locations)

        # Resample if effective sample size is low
        if self._effective_sample_size() < self.num_particles / 2:
            self.resample()

    def _update_beliefs_from_observation(
        self,
        episode_step: EpisodeStep,
        true_locations: Dict[str, ObjectLocation],
    ) -> None:
        """Update particle beliefs based on what human observes.

        When the human can see an object, update beliefs with some noise.
        This models that observation isn't perfect.
        """
        human_visible = getattr(episode_step, 'visible_objects_h', None)
        if human_visible is None:
            human_visible = []
        elif hasattr(human_visible, 'tolist'):
            human_visible = human_visible.tolist()

        for obj_id in human_visible:
            if obj_id in true_locations:
                true_loc = true_locations[obj_id]

                # Track that human has observed this object
                self._observed_objects[obj_id] = true_loc

                # Update particles with observation noise
                for particle in self.particles:
                    # With probability (1 - noise), update to true location
                    # With probability noise, keep old belief or update incorrectly
                    if self.rng.random() > self.observation_noise:
                        particle.object_locations[obj_id] = true_loc
                    else:
                        # Small chance of noisy observation (keep old or random)
                        if self.rng.random() < 0.5 and obj_id in particle.object_locations:
                            pass  # Keep old belief
                        else:
                            # Random nearby location
                            noisy_x = true_loc.position[0] + self.rng.uniform(-2, 2)
                            noisy_y = true_loc.position[1] + self.rng.uniform(-2, 2)
                            particle.object_locations[obj_id] = ObjectLocation(
                                object_id=obj_id,
                                container_id=true_loc.container_id,
                                room_id=true_loc.room_id,
                                position=(noisy_x, noisy_y, 0.0),
                            )

    def resample(self) -> None:
        """Resample particles using systematic resampling."""
        if not self.particles:
            return

        weights = np.array([p.weight for p in self.particles])
        cumulative = np.cumsum(weights)

        new_particles: List[Particle] = []
        step = 1.0 / self.num_particles
        u = self.rng.random() * step

        for i in range(self.num_particles):
            idx = np.searchsorted(cumulative, u, side="right")
            idx = min(idx, len(self.particles) - 1)

            old_particle = self.particles[idx]
            new_particles.append(
                Particle(
                    goal_id=old_particle.goal_id,
                    object_locations={k: v for k, v in old_particle.object_locations.items()},
                    weight=1.0 / self.num_particles,
                )
            )
            u += step

        self.particles = new_particles
        self._normalize_weights()

    def predict(self, observation: Observation) -> None:
        """Predict step: add drift to unobserved object beliefs.

        Objects that haven't been observed recently may have moved.
        """
        # Add small probability of location drift for unobserved objects
        for particle in self.particles:
            for obj_id, obj_loc in particle.object_locations.items():
                if obj_id not in self._observed_objects:
                    # Unobserved object - slight chance of drift
                    if self.rng.random() < 0.02:  # 2% chance per step
                        # Drift to nearby location
                        new_x = obj_loc.position[0] + self.rng.uniform(-1, 1)
                        new_y = obj_loc.position[1] + self.rng.uniform(-1, 1)
                        particle.object_locations[obj_id] = ObjectLocation(
                            object_id=obj_id,
                            container_id=obj_loc.container_id,
                            room_id=obj_loc.room_id,
                            position=(new_x, new_y, 0.0),
                        )

    def get_belief_distribution(self) -> Dict[str, float]:
        """Get goal distribution by marginalizing over object locations."""
        goal_counts: Dict[str, float] = {}
        for particle in self.particles:
            goal_counts[particle.goal_id] = (
                goal_counts.get(particle.goal_id, 0.0) + particle.weight
            )

        total = sum(goal_counts.values())
        if total > 0:
            return {goal_id: count / total for goal_id, count in goal_counts.items()}
        else:
            num_goals = len(self.task_list)
            return {
                task.task_id: 1.0 / num_goals if num_goals > 0 else 0.0
                for task in self.task_list
            }

    def get_object_location_beliefs(self) -> Dict[str, Dict[str, float]]:
        """Get object location beliefs by marginalizing over goals.

        Returns:
            Dictionary mapping object_id to {room_id: probability}
        """
        object_beliefs: Dict[str, Dict[str, float]] = {}

        for particle in self.particles:
            for obj_id, obj_loc in particle.object_locations.items():
                if obj_id not in object_beliefs:
                    object_beliefs[obj_id] = {}

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
        """Get most likely goal (MAP estimate)."""
        distribution = self.get_belief_distribution()
        if not distribution:
            return None
        return max(distribution.items(), key=lambda x: x[1])[0]

    def get_most_likely_locations(self) -> Dict[str, ObjectLocation]:
        """Get most likely object locations (MAP estimate)."""
        # Aggregate by location and find most likely
        location_counts: Dict[str, Dict[str, float]] = {}
        location_objects: Dict[str, Dict[str, ObjectLocation]] = {}

        for particle in self.particles:
            for obj_id, obj_loc in particle.object_locations.items():
                if obj_id not in location_counts:
                    location_counts[obj_id] = {}
                    location_objects[obj_id] = {}

                key = f"{obj_loc.room_id}_{obj_loc.container_id}"
                location_counts[obj_id][key] = (
                    location_counts[obj_id].get(key, 0.0) + particle.weight
                )
                location_objects[obj_id][key] = obj_loc

        most_likely = {}
        for obj_id in location_counts:
            if location_counts[obj_id]:
                best_key = max(location_counts[obj_id].items(), key=lambda x: x[1])[0]
                most_likely[obj_id] = location_objects[obj_id][best_key]

        return most_likely

    def reset(self) -> None:
        """Reset particle filter to initial prior distribution."""
        self._observed_objects.clear()
        self._initialize_particles()

    def _normalize_weights(self) -> None:
        """Normalize particle weights to sum to 1."""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for particle in self.particles:
                particle.weight /= total
        else:
            uniform_weight = 1.0 / len(self.particles) if self.particles else 0.0
            for particle in self.particles:
                particle.weight = uniform_weight

    def _effective_sample_size(self) -> float:
        """Compute effective sample size for resampling decision."""
        if not self.particles:
            return 0.0

        weights = np.array([p.weight for p in self.particles])
        weights_squared = weights**2
        ess = 1.0 / np.sum(weights_squared) if np.sum(weights_squared) > 0 else 0.0
        return ess
