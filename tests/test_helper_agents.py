"""Tests for helper agents."""

import pytest

from src.bsa.agents.helper.reactive import ReactiveHelper
from src.bsa.agents.helper.goal_only import GoalOnlyHelper
from src.bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper
from src.bsa.agents.helper.policies import InterventionPolicy
from src.bsa.agents.helper.base import HelperAgent
from src.bsa.common.types import Action, Observation, EpisodeStep, ObjectLocation
from src.bsa.envs.gridhouse import GridHouseEnvironment
from src.bsa.envs.gridhouse.tasks import get_task, list_tasks


@pytest.fixture
def sample_observation():
    """Fixture for sample observation."""
    return Observation(
        agent_id="helper",
        visible_objects=["knife"],
        visible_containers=["cabinet"],
        current_room="kitchen",
        position=(1.0, 0.0, 1.0),
        timestamp=0,
    )


@pytest.fixture
def sample_episode_step():
    """Fixture for sample episode step."""
    return EpisodeStep(
        episode_id="ep_0",
        timestep=0,
        human_action=Action.PICKUP,
        helper_obs=Observation(
            agent_id="helper",
            visible_objects=["knife"],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        ),
        human_obs=Observation(
            agent_id="human",
            visible_objects=["knife"],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        ),
        visible_objects_h=["knife"],
        visible_objects_helper=["knife"],
        true_object_locations={
            "knife": ObjectLocation(
                object_id="knife",
                container_id=None,
                room_id="kitchen",
                position=(1.0, 0.0, 1.0),
            ),
        },
        human_belief_object_locations={
            "knife": ObjectLocation(
                object_id="knife",
                container_id=None,
                room_id="kitchen",
                position=(1.0, 0.0, 1.0),
            ),
        },
        goal_id="prepare_meal",
        tau=None,
        intervention_type=None,
    )


@pytest.fixture
def sample_tasks():
    """Fixture for sample tasks."""
    return [get_task(task_id) for task_id in list_tasks()]


@pytest.fixture
def gridhouse_env():
    """Fixture for GridHouseEnvironment."""
    return GridHouseEnvironment(seed=42)


class TestReactiveHelper:
    """Tests for ReactiveHelper."""

    def test_implements_interface(self):
        """Verify implements HelperAgent interface."""
        helper = ReactiveHelper()
        assert isinstance(helper, HelperAgent)

    def test_plan_action(self, sample_observation):
        """Test reactive action planning (reacts to visible objects)."""
        helper = ReactiveHelper()
        
        # With visible objects, should pick up
        action = helper.plan_action(sample_observation)
        assert action in [Action.PICKUP, Action.OPEN, Action.MOVE]
        
        # With only containers visible
        obs_containers = Observation(
            agent_id="helper",
            visible_objects=[],
            visible_containers=["cabinet"],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        )
        action = helper.plan_action(obs_containers)
        assert action == Action.OPEN
        
        # With nothing visible, should move
        obs_empty = Observation(
            agent_id="helper",
            visible_objects=[],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        )
        action = helper.plan_action(obs_empty)
        assert action == Action.MOVE

    def test_update_belief(self, sample_observation):
        """Test that update_belief is no-op (reactive doesn't track beliefs)."""
        helper = ReactiveHelper()
        
        # Should not raise error
        helper.update_belief(sample_observation, Action.PICKUP)
        
        # Belief state should still be None
        assert helper.get_belief_state() is None

    def test_get_belief_state(self):
        """Test returns None (no belief state)."""
        helper = ReactiveHelper()
        assert helper.get_belief_state() is None

    def test_detect_false_belief(self, sample_observation):
        """Test returns False (reactive doesn't detect)."""
        helper = ReactiveHelper()
        assert helper.detect_false_belief(sample_observation) is False

    def test_reset(self):
        """Test agent reset."""
        helper = ReactiveHelper()
        # Reset should not raise error (no state to reset)
        helper.reset()


class TestGoalOnlyHelper:
    """Tests for GoalOnlyHelper."""

    def test_implements_interface(self, sample_tasks):
        """Verify implements HelperAgent interface."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        assert isinstance(helper, HelperAgent)

    def test_goal_inference(self, sample_tasks, sample_observation, sample_episode_step):
        """Test goal inference updates correctly."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        
        initial_dist = helper.get_belief_state()["goal_distribution"]
        
        # Update with action
        helper.update_belief(sample_observation, Action.PICKUP, sample_episode_step)
        
        updated_dist = helper.get_belief_state()["goal_distribution"]
        
        # Distribution should have changed
        assert updated_dist != initial_dist

    def test_plan_action(self, sample_tasks, sample_observation, sample_episode_step):
        """Test action planning based on inferred goal."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        
        # Plan action
        action = helper.plan_action(sample_observation, sample_episode_step)
        
        assert isinstance(action, Action)

    def test_update_belief(self, sample_tasks, sample_observation, sample_episode_step):
        """Test updates goal inference."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        
        initial_dist = helper.get_belief_state()["goal_distribution"]
        
        helper.update_belief(sample_observation, Action.PICKUP, sample_episode_step)
        
        updated_dist = helper.get_belief_state()["goal_distribution"]
        assert updated_dist != initial_dist

    def test_get_belief_state(self, sample_tasks):
        """Test returns goal distribution."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        
        belief_state = helper.get_belief_state()
        
        assert isinstance(belief_state, dict)
        assert "goal_distribution" in belief_state
        assert isinstance(belief_state["goal_distribution"], dict)

    def test_detect_false_belief(self, sample_tasks, sample_observation):
        """Test returns False (assumes beliefs match true state)."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        assert helper.detect_false_belief(sample_observation) is False

    def test_reset(self, sample_tasks):
        """Test agent reset."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        
        # Update beliefs
        obs = Observation(
            agent_id="helper",
            visible_objects=["knife"],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        )
        helper.update_belief(obs, Action.PICKUP)
        
        initial_dist = helper.get_belief_state()["goal_distribution"]
        
        # Reset
        helper.reset()
        
        reset_dist = helper.get_belief_state()["goal_distribution"]
        
        # Should match initial distribution
        assert reset_dist == initial_dist


class TestBeliefSensitiveHelper:
    """Tests for BeliefSensitiveHelper."""

    def test_implements_interface(self, sample_tasks):
        """Verify implements HelperAgent interface."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        assert isinstance(helper, HelperAgent)

    def test_belief_tracking(self, sample_tasks, sample_observation, sample_episode_step):
        """Test tracks both goal and object location beliefs."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        # Update beliefs
        helper.update_belief(
            sample_observation,
            Action.PICKUP,
            sample_episode_step,
        )
        
        belief_state = helper.get_belief_state()
        
        assert "goal_distribution" in belief_state
        assert "object_location_beliefs" in belief_state

    def test_plan_action(self, sample_tasks, sample_observation, sample_episode_step):
        """Test action planning based on beliefs."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        action = helper.plan_action(sample_observation, sample_episode_step)
        
        assert isinstance(action, Action)

    def test_update_belief(self, sample_tasks, sample_observation, sample_episode_step):
        """Test updates belief inference."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        initial_state = helper.get_belief_state()
        
        helper.update_belief(
            sample_observation,
            Action.PICKUP,
            sample_episode_step,
        )
        
        updated_state = helper.get_belief_state()
        assert updated_state != initial_state

    def test_detect_false_belief(self, sample_tasks, sample_observation, sample_episode_step):
        """Test false-belief detection logic."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        # Update beliefs
        helper.update_belief(
            sample_observation,
            Action.PICKUP,
            sample_episode_step,
        )
        
        # Detect false belief
        false_belief = helper.detect_false_belief(sample_observation, sample_episode_step)
        
        assert isinstance(false_belief, bool)

    def test_get_belief_state(self, sample_tasks):
        """Test returns full belief state."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        belief_state = helper.get_belief_state()
        
        assert isinstance(belief_state, dict)
        assert "goal_distribution" in belief_state
        assert "object_location_beliefs" in belief_state

    def test_intervention_policy(self, sample_tasks, sample_observation, sample_episode_step):
        """Test intervention policy decisions."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        # Plan action (uses intervention policy)
        action = helper.plan_action(sample_observation, sample_episode_step)
        
        assert isinstance(action, Action)

    def test_reset(self, sample_tasks, sample_observation, sample_episode_step):
        """Test agent reset."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        
        # Update beliefs
        helper.update_belief(
            sample_observation,
            Action.PICKUP,
            sample_episode_step,
        )
        
        initial_state = helper.get_belief_state()
        
        # Reset
        helper.reset()
        
        reset_state = helper.get_belief_state()
        
        # Goal distribution should match initial
        assert reset_state["goal_distribution"] == initial_state["goal_distribution"]


class TestInterventionPolicy:
    """Tests for InterventionPolicy."""

    def test_should_intervene(self, sample_tasks):
        """Test intervention decision logic."""
        policy = InterventionPolicy()
        task = sample_tasks[0]
        
        # Test with false belief
        belief_state = {
            "goal_distribution": {"prepare_meal": 0.8, "find_keys": 0.2},
            "object_location_beliefs": {
                "knife": {"kitchen": 0.9, "living_room": 0.1},
            },
        }
        true_locations = {
            "knife": ObjectLocation(
                object_id="knife",
                container_id=None,
                room_id="living_room",  # Different from belief
                position=(1.0, 0.0, 1.0),
            ),
        }
        observation = Observation(
            agent_id="helper",
            visible_objects=[],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        )
        
        should_intervene = policy.should_intervene(
            belief_state, true_locations, observation, task
        )
        
        assert isinstance(should_intervene, bool)

    def test_choose_intervention(self, sample_tasks):
        """Test intervention type selection."""
        policy = InterventionPolicy()
        task = sample_tasks[0]
        
        belief_state = {
            "goal_distribution": {"prepare_meal": 0.8},
            "object_location_beliefs": {},
        }
        true_locations = {
            "knife": ObjectLocation(
                object_id="knife",
                container_id=None,
                room_id="kitchen",
                position=(1.0, 0.0, 1.0),
            ),
        }
        observation = Observation(
            agent_id="helper",
            visible_objects=["knife"],
            visible_containers=[],
            current_room="kitchen",
            position=(1.0, 0.0, 1.0),
            timestamp=0,
        )
        
        action = policy.choose_intervention(
            belief_state, true_locations, observation, task
        )
        
        assert isinstance(action, Action)

    def test_get_intervention_type(self, sample_tasks):
        """Test intervention type retrieval."""
        policy = InterventionPolicy()
        
        belief_state = {
            "goal_distribution": {"prepare_meal": 0.8},
            "object_location_beliefs": {},
        }
        true_locations = {
            "knife": ObjectLocation(
                object_id="knife",
                container_id=None,
                room_id="kitchen",
                position=(1.0, 0.0, 1.0),
            ),
        }
        
        intervention_type = policy.get_intervention_type(belief_state, true_locations)
        
        assert isinstance(intervention_type, str)
        assert intervention_type in ["communicate", "assist"]


class TestHelperAgentIntegration:
    """Integration tests for helper agents with environments."""

    def test_reactive_helper_with_gridhouse(self, gridhouse_env):
        """Test ReactiveHelper works with GridHouseEnvironment."""
        helper = ReactiveHelper()
        gridhouse_env.reset(seed=42)
        
        obs = gridhouse_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        
        assert isinstance(action, Action)

    def test_goal_only_helper_with_gridhouse(self, gridhouse_env, sample_tasks):
        """Test GoalOnlyHelper works with GridHouseEnvironment."""
        helper = GoalOnlyHelper(sample_tasks, seed=42)
        gridhouse_env.reset(seed=42)
        
        obs = gridhouse_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        
        assert isinstance(action, Action)

    def test_belief_sensitive_helper_with_gridhouse(self, gridhouse_env, sample_tasks):
        """Test BeliefSensitiveHelper works with GridHouseEnvironment."""
        helper = BeliefSensitiveHelper(sample_tasks, num_particles=100, seed=42)
        gridhouse_env.reset(seed=42)
        
        obs = gridhouse_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        
        assert isinstance(action, Action)
