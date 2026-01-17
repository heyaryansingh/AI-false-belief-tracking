"""Unit tests for VirtualHome components."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import VirtualHome components
try:
    from src.bsa.envs.virtualhome import (
        VirtualHomeEnvironment,
        VirtualHomeEpisodeGenerator,
        VirtualHomeEpisodeRecorder,
        get_scene_state,
        get_agent_view,
    )
    from src.bsa.common.types import Action
    VIRTUALHOME_AVAILABLE = True
except ImportError:
    VIRTUALHOME_AVAILABLE = False
    pytest.skip("VirtualHome not installed - skipping tests", allow_module_level=True)


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


class TestVirtualHomeEnvironment:
    """Tests for VirtualHomeEnvironment."""

    def test_reset(self, virtualhome_env):
        """Test environment reset."""
        obs = virtualhome_env.reset(seed=42)
        assert obs is not None
        assert obs.agent_id == "human"
        assert obs.current_room is not None

    def test_step(self, virtualhome_env):
        """Test environment step."""
        obs, reward, done, info = virtualhome_env.step(Action.MOVE, "human")
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_get_visible_state(self, virtualhome_env):
        """Test partial observability."""
        obs = virtualhome_env.get_visible_state("human")
        assert obs.agent_id == "human"
        assert isinstance(obs.visible_objects, list)
        assert isinstance(obs.visible_containers, list)
        assert obs.current_room is not None

    def test_get_object_locations(self, virtualhome_env):
        """Test object location tracking."""
        locations = virtualhome_env.get_object_locations()
        assert isinstance(locations, dict)
        # Locations may be empty in stub implementation

    def test_get_true_state(self, virtualhome_env):
        """Test getting true state."""
        state = virtualhome_env.get_true_state()
        assert isinstance(state, dict)
        assert "object_locations" in state
        assert "agent_positions" in state
        assert "timestep" in state


class TestVirtualHomeEpisodeGenerator:
    """Tests for VirtualHomeEpisodeGenerator."""

    @pytest.fixture
    def generator(self, virtualhome_env):
        """Create episode generator fixture."""
        if not VIRTUALHOME_AVAILABLE:
            pytest.skip("VirtualHome not installed")
        return VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)

    def test_generate_episode(self, generator):
        """Test episode generation."""
        episode = generator.generate_episode(goal_id="prepare_meal", tau=5)
        assert episode is not None
        assert episode.goal_id == "prepare_meal"
        assert len(episode.steps) > 0
        assert episode.tau == 5

    def test_episode_structure(self, generator):
        """Test episode has correct structure."""
        episode = generator.generate_episode()
        assert hasattr(episode, "episode_id")
        assert hasattr(episode, "goal_id")
        assert hasattr(episode, "steps")
        assert hasattr(episode, "metadata")
        assert isinstance(episode.steps, list)
        assert len(episode.steps) > 0

        # Check step structure
        step = episode.steps[0]
        assert hasattr(step, "timestep")
        assert hasattr(step, "human_action")
        assert hasattr(step, "human_obs")
        assert hasattr(step, "helper_obs")
        assert hasattr(step, "true_object_locations")
        assert hasattr(step, "human_belief_object_locations")

    def test_intervention(self, generator):
        """Test false-belief intervention."""
        episode = generator.generate_episode(goal_id="prepare_meal", tau=5, intervention_type="relocate")
        assert episode.intervention_type == "relocate"
        assert episode.tau == 5

        # Check that intervention was applied (if episode long enough)
        if len(episode.steps) > 5:
            step_after = episode.steps[5]
            # Intervention should be marked
            assert step_after.tau == 5

    def test_belief_tracking(self, generator):
        """Test human agent belief tracking."""
        episode = generator.generate_episode(goal_id="prepare_meal")
        
        # Check that beliefs are tracked
        for step in episode.steps:
            assert isinstance(step.human_belief_object_locations, dict)
            assert isinstance(step.true_object_locations, dict)


class TestVirtualHomeEpisodeRecorder:
    """Tests for VirtualHomeEpisodeRecorder."""

    @pytest.fixture
    def episode(self, virtualhome_env):
        """Create test episode fixture."""
        if not VIRTUALHOME_AVAILABLE:
            pytest.skip("VirtualHome not installed")
        generator = VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)
        return generator.generate_episode(goal_id="prepare_meal")

    def test_save_parquet(self, episode, tmp_path):
        """Test Parquet serialization."""
        recorder = VirtualHomeEpisodeRecorder()
        output_path = tmp_path / "test_episode.parquet"
        recorder.save_episode(episode, output_path, format="parquet")
        assert output_path.exists()

    def test_save_jsonl(self, episode, tmp_path):
        """Test JSONL serialization."""
        recorder = VirtualHomeEpisodeRecorder()
        output_path = tmp_path / "test_episode.jsonl"
        recorder.save_episode(episode, output_path, format="jsonl")
        assert output_path.exists()


class TestObservabilityModule:
    """Tests for observability module."""

    def test_get_scene_state(self, virtualhome_env):
        """Test scene state queries."""
        state = get_scene_state(virtualhome_env)
        assert isinstance(state, dict)
        assert "rooms" in state
        assert "objects" in state
        assert "agents" in state

    def test_get_agent_view(self, virtualhome_env):
        """Test agent view queries."""
        view = get_agent_view(virtualhome_env, "human")
        assert isinstance(view, dict)
        assert "agent_id" in view
        assert "current_room" in view
        assert "visible_objects" in view
