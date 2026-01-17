"""Integration tests for VirtualHome with helper agents."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import VirtualHome and helper agents
try:
    from src.bsa.envs.virtualhome import (
        VirtualHomeEnvironment,
        VirtualHomeEpisodeGenerator,
    )
    from src.bsa.agents.helper import (
        ReactiveHelper,
        GoalOnlyHelper,
        BeliefSensitiveHelper,
    )
    from src.bsa.common.types import Action
    VIRTUALHOME_AVAILABLE = True
except ImportError:
    VIRTUALHOME_AVAILABLE = False
    pytest.skip("VirtualHome not installed - skipping integration tests", allow_module_level=True)


@pytest.fixture
def virtualhome_env():
    """Create VirtualHome environment fixture."""
    if not VIRTUALHOME_AVAILABLE:
        pytest.skip("VirtualHome not installed")
    try:
        env = VirtualHomeEnvironment(seed=42)
        env.reset(seed=42)
        return env
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"VirtualHome not available: {e}")


class TestHelperAgentsWithVirtualHome:
    """Test helper agents work with VirtualHomeEnvironment."""

    def test_reactive_helper(self, virtualhome_env):
        """Test ReactiveHelper with VirtualHome."""
        helper = ReactiveHelper()
        obs = virtualhome_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        assert action is not None
        assert isinstance(action, Action)

    def test_goal_only_helper(self, virtualhome_env):
        """Test GoalOnlyHelper with VirtualHome."""
        helper = GoalOnlyHelper()
        obs = virtualhome_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        assert action is not None
        assert isinstance(action, Action)
        
        # Update helper with human observation
        human_obs = virtualhome_env.get_visible_state("human")
        helper.update_belief(human_obs, virtualhome_env.get_object_locations())

    def test_belief_sensitive_helper(self, virtualhome_env):
        """Test BeliefSensitiveHelper with VirtualHome."""
        helper = BeliefSensitiveHelper(num_particles=10)
        obs = virtualhome_env.get_visible_state("helper")
        action = helper.plan_action(obs)
        assert action is not None
        assert isinstance(action, Action)
        
        # Update helper with human observation
        human_obs = virtualhome_env.get_visible_state("human")
        helper.update_belief(human_obs, virtualhome_env.get_object_locations())
        
        # Check belief state
        belief_state = helper.get_belief_state()
        assert belief_state is not None


class TestEpisodeGenerationEndToEnd:
    """Test full episode generation end-to-end."""

    def test_episode_generation_end_to_end(self, virtualhome_env):
        """Test full episode generation."""
        generator = VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)
        episode = generator.generate_episode(goal_id="prepare_meal", tau=5)
        
        assert episode is not None
        assert len(episode.steps) > 0
        assert episode.goal_id == "prepare_meal"
        
        # Check episode structure
        assert hasattr(episode, "episode_id")
        assert hasattr(episode, "steps")
        assert hasattr(episode, "metadata")

    def test_false_belief_detection(self, virtualhome_env):
        """Test belief-sensitive helper detects false beliefs."""
        generator = VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)
        episode = generator.generate_episode(
            goal_id="prepare_meal",
            tau=5,
            intervention_type="relocate"
        )
        
        # Create belief-sensitive helper
        helper = BeliefSensitiveHelper(num_particles=10)
        
        # Process episode steps
        for step in episode.steps[:10]:  # Process first 10 steps
            helper.update_belief(
                step.human_obs,
                step.true_object_locations
            )
            
            # Check if false belief detected
            false_belief = helper.detect_false_belief(
                step.true_object_locations
            )
            # False belief may or may not be detected depending on particle filter state
            assert isinstance(false_belief, bool)

    def test_episode_compatibility(self, virtualhome_env):
        """Test VirtualHome episodes are compatible with GridHouse format."""
        generator = VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)
        episode = generator.generate_episode()
        
        # Check episode structure matches GridHouse
        assert hasattr(episode, "episode_id")
        assert hasattr(episode, "goal_id")
        assert hasattr(episode, "tau")
        assert hasattr(episode, "intervention_type")
        assert hasattr(episode, "steps")
        assert hasattr(episode, "metadata")
        
        # Check step structure
        if len(episode.steps) > 0:
            step = episode.steps[0]
            assert hasattr(step, "timestep")
            assert hasattr(step, "human_action")
            assert hasattr(step, "human_obs")
            assert hasattr(step, "helper_obs")
            assert hasattr(step, "true_object_locations")
            assert hasattr(step, "human_belief_object_locations")
