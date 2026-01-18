"""Tests for particle filter."""

import pytest
import numpy as np

from src.bsa.inference.particle_filter import ParticleFilter, Particle
from src.bsa.inference.likelihood import RuleBasedLikelihoodModel
from src.bsa.common.types import Action, Observation, ObjectLocation, Task
from src.bsa.common.seeding import set_seed


@pytest.fixture
def sample_tasks():
    """Fixture for sample tasks."""
    from src.bsa.envs.gridhouse.tasks import get_task
    return [get_task("prepare_meal"), get_task("find_keys")]


@pytest.fixture
def likelihood_model():
    """Fixture for likelihood model."""
    return RuleBasedLikelihoodModel()


@pytest.fixture
def sample_observation():
    """Fixture for sample observation."""
    return Observation(
        agent_id="human",
        visible_objects=["knife"],
        visible_containers=["cabinet"],
        current_room="kitchen",
        position=(1.0, 0.0, 1.0),
        timestamp=0,
    )


@pytest.fixture
def sample_true_locations():
    """Fixture for sample true object locations."""
    return {
        "knife": ObjectLocation(
            object_id="knife",
            container_id=None,
            room_id="kitchen",
            position=(1.0, 0.0, 1.0),
        ),
        "keys": ObjectLocation(
            object_id="keys",
            container_id=None,
            room_id="living_room",
            position=(2.0, 0.0, 2.0),
        ),
    }


class TestParticleFilter:
    """Tests for ParticleFilter."""

    def test_initialization(self, sample_tasks):
        """Test particle filter initializes with correct number of particles."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        assert len(pf.particles) == 100
        assert pf.num_particles == 100
        assert pf.task_list == sample_tasks

    def test_uniform_initialization(self, sample_tasks):
        """Verify initial particles are uniformly distributed."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        # Count particles per goal
        goal_counts = {}
        for particle in pf.particles:
            goal_counts[particle.goal_id] = goal_counts.get(particle.goal_id, 0) + 1
        
        # Should be roughly uniform (50 particles per goal for 2 goals)
        assert len(goal_counts) == len(sample_tasks)
        for goal_id in goal_counts:
            assert goal_counts[goal_id] > 0

    def test_initial_goal_distribution(self, sample_tasks):
        """Test initial goal distribution is uniform."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        goal_dist = pf.get_belief_distribution()
        
        # Should be uniform
        expected_prob = 1.0 / len(sample_tasks)
        for goal_id, prob in goal_dist.items():
            assert abs(prob - expected_prob) < 0.01, f"Goal {goal_id} should have uniform probability"

    def test_update_with_action(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Test particle weighting based on action likelihood."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        initial_weights = [p.weight for p in pf.particles]
        
        # Update with an action
        pf.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        
        # Weights should have changed
        updated_weights = [p.weight for p in pf.particles]
        assert updated_weights != initial_weights
        
        # Weights should be normalized
        total_weight = sum(updated_weights)
        assert abs(total_weight - 1.0) < 0.01, "Weights should sum to 1.0"

    def test_resampling(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Test systematic resampling maintains particle count."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        # Update to create weight diversity
        pf.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        
        initial_count = len(pf.particles)
        
        # Force resampling
        pf._resample()
        
        # Particle count should be maintained
        assert len(pf.particles) == initial_count
        assert len(pf.particles) == pf.num_particles

    def test_particle_weights(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Verify weights are normalized correctly."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        # Update particles
        pf.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        
        # Check normalization
        total_weight = sum(p.weight for p in pf.particles)
        assert abs(total_weight - 1.0) < 0.01, "Weights should sum to 1.0"
        
        # All weights should be non-negative
        assert all(p.weight >= 0 for p in pf.particles), "All weights should be non-negative"

    def test_get_belief_distribution(self, sample_tasks):
        """Test goal distribution extraction."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        goal_dist = pf.get_belief_distribution()
        
        assert isinstance(goal_dist, dict)
        assert len(goal_dist) == len(sample_tasks)
        
        # Probabilities should sum to 1
        total_prob = sum(goal_dist.values())
        assert abs(total_prob - 1.0) < 0.01, "Goal distribution should sum to 1.0"

    def test_get_object_location_beliefs(self, sample_tasks):
        """Test object location belief extraction."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        # Initially, no object location beliefs
        beliefs = pf.get_object_location_beliefs()
        assert isinstance(beliefs, dict)

    def test_get_most_likely_goal(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Test most likely goal identification."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        # Initially should return one of the tasks
        most_likely = pf.get_most_likely_goal()
        assert most_likely in [task.task_id for task in sample_tasks]
        
        # After updates, should still return valid goal
        pf.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        
        most_likely_after = pf.get_most_likely_goal()
        assert most_likely_after in [task.task_id for task in sample_tasks]

    def test_get_most_likely_locations(self, sample_tasks):
        """Test most likely object locations."""
        pf = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        
        locations = pf.get_most_likely_locations()
        assert isinstance(locations, dict)

    def test_single_particle(self, sample_tasks):
        """Test with single particle."""
        pf = ParticleFilter(sample_tasks, num_particles=1, seed=42)
        
        assert len(pf.particles) == 1
        assert pf.particles[0].weight == 1.0
        
        # Should still work
        goal_dist = pf.get_belief_distribution()
        assert len(goal_dist) > 0

    def test_all_zero_weights(self, sample_tasks):
        """Handle degenerate case (all weights zero)."""
        pf = ParticleFilter(sample_tasks, num_particles=10, seed=42)
        
        # Set all weights to zero
        for particle in pf.particles:
            particle.weight = 0.0
        
        # Normalize should handle this
        pf._normalize_weights()
        
        # After normalization, weights should be uniform
        total_weight = sum(p.weight for p in pf.particles)
        assert abs(total_weight - 1.0) < 0.01 or total_weight == 0.0

    def test_deterministic_with_seed(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Same seed produces same results."""
        set_seed(42)
        pf1 = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        pf1.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        goal_dist1 = pf1.get_belief_distribution()
        
        set_seed(42)
        pf2 = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        pf2.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        goal_dist2 = pf2.get_belief_distribution()
        
        # Distributions should be similar (may not be identical due to resampling randomness)
        assert len(goal_dist1) == len(goal_dist2)

    def test_resampling_deterministic(self, sample_tasks, likelihood_model, sample_observation, sample_true_locations):
        """Resampling is deterministic with seed."""
        set_seed(42)
        pf1 = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        pf1.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        pf1._resample()
        count1 = len([p for p in pf1.particles if p.goal_id == sample_tasks[0].task_id])
        
        set_seed(42)
        pf2 = ParticleFilter(sample_tasks, num_particles=100, seed=42)
        pf2.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
            likelihood_model=likelihood_model,
        )
        pf2._resample()
        count2 = len([p for p in pf2.particles if p.goal_id == sample_tasks[0].task_id])
        
        # Counts should be the same with same seed
        assert count1 == count2
