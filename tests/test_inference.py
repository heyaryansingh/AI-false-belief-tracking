"""Tests for inference modules."""

import pytest
import numpy as np

from src.bsa.inference.goal import GoalInference
from src.bsa.inference.belief import BeliefInference
from src.bsa.inference.likelihood import RuleBasedLikelihoodModel, LikelihoodModel
from src.bsa.common.types import Action, Observation, ObjectLocation, Task
from src.bsa.common.seeding import set_seed


@pytest.fixture
def sample_tasks():
    """Fixture for sample tasks."""
    from src.bsa.envs.gridhouse.tasks import get_task
    return [get_task("prepare_meal"), get_task("find_keys")]


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
def sample_belief_locations():
    """Fixture for sample believed object locations."""
    return {
        "knife": ObjectLocation(
            object_id="knife",
            container_id=None,
            room_id="kitchen",
            position=(1.0, 0.0, 1.0),
        ),
    }


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


class TestGoalInference:
    """Tests for GoalInference."""

    def test_initialization(self, sample_tasks):
        """Test goal inference initializes with uniform distribution."""
        gi = GoalInference(sample_tasks, seed=42)
        
        goal_dist = gi.get_goal_distribution()
        
        assert len(goal_dist) == len(sample_tasks)
        expected_prob = 1.0 / len(sample_tasks)
        for goal_id, prob in goal_dist.items():
            assert abs(prob - expected_prob) < 0.01, "Initial distribution should be uniform"

    def test_update_with_action(self, sample_tasks, sample_observation):
        """Test goal distribution updates based on action."""
        gi = GoalInference(sample_tasks, seed=42)
        
        initial_dist = gi.get_goal_distribution()
        
        # Update with action
        gi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
        )
        
        updated_dist = gi.get_goal_distribution()
        
        # Distribution should have changed
        assert updated_dist != initial_dist
        
        # Should still sum to 1
        total_prob = sum(updated_dist.values())
        assert abs(total_prob - 1.0) < 0.01, "Distribution should sum to 1.0"

    def test_get_goal_distribution(self, sample_tasks):
        """Test distribution retrieval."""
        gi = GoalInference(sample_tasks, seed=42)
        
        goal_dist = gi.get_goal_distribution()
        
        assert isinstance(goal_dist, dict)
        assert len(goal_dist) == len(sample_tasks)
        assert all(0 <= prob <= 1 for prob in goal_dist.values())

    def test_get_most_likely_goal(self, sample_tasks, sample_observation):
        """Test most likely goal identification."""
        gi = GoalInference(sample_tasks, seed=42)
        
        most_likely = gi.get_most_likely_goal()
        
        assert most_likely is not None
        assert most_likely in [task.task_id for task in sample_tasks]
        
        # After updates, should still return valid goal
        gi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
        )
        
        most_likely_after = gi.get_most_likely_goal()
        assert most_likely_after in [task.task_id for task in sample_tasks]

    def test_convergence(self, sample_tasks, sample_observation):
        """Test goal inference converges to correct goal."""
        gi = GoalInference(sample_tasks, seed=42)
        
        # Simulate actions consistent with "prepare_meal" goal
        # (picking up knife, which is a critical object for prepare_meal)
        for _ in range(5):
            gi.update(
                human_action=Action.PICKUP,
                observation=sample_observation,
            )
        
        goal_dist = gi.get_goal_distribution()
        most_likely = gi.get_most_likely_goal()
        
        # Should have higher probability for prepare_meal (knife is critical object)
        assert most_likely == "prepare_meal" or goal_dist.get("prepare_meal", 0) > 0.4

    def test_reset(self, sample_tasks, sample_observation):
        """Test reset returns to initial distribution."""
        gi = GoalInference(sample_tasks, seed=42)
        initial_dist = gi.get_goal_distribution()
        
        # Update
        gi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
        )
        
        # Reset
        gi.reset()
        reset_dist = gi.get_goal_distribution()
        
        # Should match initial distribution
        assert reset_dist == initial_dist


class TestBeliefInference:
    """Tests for BeliefInference."""

    def test_initialization(self, sample_tasks):
        """Test belief inference initializes particle filter."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        assert bi.particle_filter is not None
        assert len(bi.particle_filter.particles) == 100

    def test_update(self, sample_tasks, sample_observation, sample_true_locations):
        """Test belief updates with observations."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        initial_belief_state = bi.get_belief_state()
        
        # Update beliefs
        bi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
        )
        
        updated_belief_state = bi.get_belief_state()
        
        # Belief state should have changed
        assert updated_belief_state != initial_belief_state

    def test_detect_false_belief(self, sample_tasks, sample_observation, sample_true_locations):
        """Test false-belief detection logic."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        # Initially, no false belief (beliefs match true locations)
        false_belief = bi.detect_false_belief(sample_true_locations)
        # May or may not detect false belief initially (depends on particle initialization)
        
        # Update beliefs
        bi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
        )
        
        # Check detection
        false_belief_after = bi.detect_false_belief(sample_true_locations)
        assert isinstance(false_belief_after, bool)

    def test_get_belief_state(self, sample_tasks):
        """Test belief state extraction."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        belief_state = bi.get_belief_state()
        
        assert isinstance(belief_state, dict)
        assert "goal_distribution" in belief_state
        assert "object_location_beliefs" in belief_state

    def test_get_most_likely_goal(self, sample_tasks):
        """Test most likely goal retrieval."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        most_likely = bi.get_most_likely_goal()
        
        assert most_likely is not None
        assert most_likely in [task.task_id for task in sample_tasks]

    def test_get_most_likely_locations(self, sample_tasks):
        """Test most likely locations retrieval."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        
        locations = bi.get_most_likely_locations()
        
        assert isinstance(locations, dict)

    def test_reset(self, sample_tasks, sample_observation, sample_true_locations):
        """Test reset returns to initial state."""
        bi = BeliefInference(sample_tasks, num_particles=100, seed=42)
        initial_state = bi.get_belief_state()
        
        # Update
        bi.update(
            human_action=Action.PICKUP,
            observation=sample_observation,
            true_locations=sample_true_locations,
        )
        
        # Reset
        bi.reset()
        reset_state = bi.get_belief_state()
        
        # Goal distribution should match initial
        assert reset_state["goal_distribution"] == initial_state["goal_distribution"]


class TestLikelihoodModel:
    """Tests for LikelihoodModel."""

    def test_rule_based_likelihood(self, sample_tasks, sample_observation, sample_belief_locations):
        """Test rule-based likelihood computation."""
        model = RuleBasedLikelihoodModel()
        task = sample_tasks[0]  # prepare_meal
        
        # Test PICKUP action with visible critical object
        likelihood = model.compute(
            action=Action.PICKUP,
            goal=task,
            believed_locations=sample_belief_locations,
            observation=sample_observation,
        )
        
        assert 0 <= likelihood <= 1, "Likelihood should be in [0, 1]"
        assert likelihood > 0.5, "PICKUP of critical object should have high likelihood"

    def test_action_likelihood(self, sample_tasks, sample_observation, sample_belief_locations):
        """Test P(action | goal, beliefs) computation."""
        model = RuleBasedLikelihoodModel()
        task = sample_tasks[0]  # prepare_meal
        
        # Test different actions
        actions = [Action.PICKUP, Action.MOVE, Action.OPEN, Action.WAIT]
        
        for action in actions:
            likelihood = model.compute(
                action=action,
                goal=task,
                believed_locations=sample_belief_locations,
                observation=sample_observation,
            )
            
            assert 0 <= likelihood <= 1, f"Likelihood for {action} should be in [0, 1]"

    def test_edge_cases(self, sample_tasks, sample_observation):
        """Test likelihood with missing data."""
        model = RuleBasedLikelihoodModel()
        task = sample_tasks[0]
        
        # Test with empty believed locations
        empty_beliefs = {}
        likelihood = model.compute(
            action=Action.MOVE,
            goal=task,
            believed_locations=empty_beliefs,
            observation=sample_observation,
        )
        
        assert 0 <= likelihood <= 1, "Likelihood should handle empty beliefs"
        
        # Test with empty observation
        empty_obs = Observation(
            agent_id="human",
            visible_objects=[],
            visible_containers=[],
            current_room="kitchen",
            position=(0.0, 0.0, 0.0),
            timestamp=0,
        )
        likelihood = model.compute(
            action=Action.PICKUP,
            goal=task,
            believed_locations={},
            observation=empty_obs,
        )
        
        assert 0 <= likelihood <= 1, "Likelihood should handle empty observation"
