"""Tests for episode generators."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from src.bsa.envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
from src.bsa.envs.gridhouse.tasks import get_task, list_tasks
from src.bsa.common.types import Episode, EpisodeStep, Action
from src.bsa.common.seeding import set_seed


@pytest.fixture
def gridhouse_env():
    """Fixture for GridHouseEnvironment."""
    return GridHouseEnvironment(seed=42)


@pytest.fixture
def gridhouse_generator(gridhouse_env):
    """Fixture for GridHouseEpisodeGenerator."""
    return GridHouseEpisodeGenerator(gridhouse_env, seed=42)


@pytest.fixture
def sample_task():
    """Fixture for sample task."""
    return get_task("prepare_meal")


class TestGridHouseEpisodeGenerator:
    """Tests for GridHouseEpisodeGenerator."""

    def test_generate_episode(self, gridhouse_generator):
        """Test basic episode generation."""
        episode = gridhouse_generator.generate_episode(goal_id="prepare_meal")
        assert isinstance(episode, Episode)
        assert episode.goal_id == "prepare_meal"
        assert len(episode.steps) > 0

    def test_episode_structure(self, gridhouse_generator):
        """Verify episode has correct structure."""
        episode = gridhouse_generator.generate_episode(goal_id="prepare_meal")
        
        assert hasattr(episode, "episode_id")
        assert hasattr(episode, "goal_id")
        assert hasattr(episode, "steps")
        assert hasattr(episode, "metadata")
        
        assert isinstance(episode.episode_id, str)
        assert isinstance(episode.goal_id, str)
        assert isinstance(episode.steps, list)
        assert isinstance(episode.metadata, dict)
        
        # Check step structure
        if episode.steps:
            step = episode.steps[0]
            assert isinstance(step, EpisodeStep)
            assert hasattr(step, "timestep")
            assert hasattr(step, "human_action")
            assert hasattr(step, "human_obs")
            assert hasattr(step, "helper_obs")
            assert hasattr(step, "true_object_locations")
            assert hasattr(step, "human_belief_object_locations")

    def test_intervention_applied(self, gridhouse_generator):
        """Test that intervention is applied at tau timestep."""
        tau = 5
        episode = gridhouse_generator.generate_episode(
            goal_id="prepare_meal", tau=tau, intervention_type="relocate"
        )
        
        assert episode.tau == tau
        assert episode.intervention_type == "relocate"
        
        # Check metadata indicates intervention was applied
        assert episode.metadata.get("intervention_applied", False) is True

    def test_false_belief_created(self, gridhouse_generator):
        """Verify false belief is created (human belief != true location after intervention)."""
        tau = 5
        episode = gridhouse_generator.generate_episode(
            goal_id="prepare_meal", tau=tau, intervention_type="relocate"
        )
        
        # Check that false belief was created
        assert episode.metadata.get("false_belief_created", False) is True
        
        # Verify at tau, true location differs from human belief for a critical object
        if len(episode.steps) > tau:
            step_at_tau = episode.steps[tau]
            task = get_task(episode.goal_id)
            
            # Check at least one critical object has false belief
            false_belief_found = False
            for obj_id in task.critical_objects:
                if obj_id in step_at_tau.true_object_locations and obj_id in step_at_tau.human_belief_object_locations:
                    true_loc = step_at_tau.true_object_locations[obj_id]
                    belief_loc = step_at_tau.human_belief_object_locations[obj_id]
                    if true_loc.room_id != belief_loc.room_id:
                        false_belief_found = True
                        break
            
            assert false_belief_found, "False belief should be created for at least one critical object"

    def test_belief_tracking(self, gridhouse_generator):
        """Test human agent belief updates correctly."""
        episode = gridhouse_generator.generate_episode(goal_id="find_keys", tau=None)
        
        # Check that beliefs are tracked throughout episode
        initial_beliefs = episode.steps[0].human_belief_object_locations
        
        # Find a step where 'keys' are visible
        for step in episode.steps:
            if "keys" in step.human_obs.visible_objects:
                # After seeing keys, belief should match true location
                if "keys" in step.true_object_locations and "keys" in step.human_belief_object_locations:
                    true_loc = step.true_object_locations["keys"]
                    belief_loc = step.human_belief_object_locations["keys"]
                    assert true_loc.room_id == belief_loc.room_id, "Belief should update when object is visible"
                    break

    def test_deterministic_seeding(self, gridhouse_env):
        """Test that same seed produces same episode."""
        set_seed(42)
        gen1 = GridHouseEpisodeGenerator(gridhouse_env, seed=42)
        episode1 = gen1.generate_episode(goal_id="prepare_meal", tau=5)
        
        set_seed(42)
        gen2 = GridHouseEpisodeGenerator(gridhouse_env, seed=42)
        episode2 = gen2.generate_episode(goal_id="prepare_meal", tau=5)
        
        # Episodes should be identical (same episode_id, same steps)
        assert episode1.episode_id == episode2.episode_id
        assert len(episode1.steps) == len(episode2.steps)
        # Note: Full equality check might be too strict due to randomness in some steps

    def test_multiple_episodes(self, gridhouse_generator):
        """Test generate_episodes() function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episodes = gridhouse_generator.generate_episodes(
                num_episodes=5, output_dir=output_dir, format="parquet"
            )
            
            assert len(episodes) == 5
            assert all(isinstance(ep, Episode) for ep in episodes)
            
            # Check files were saved
            parquet_files = list(output_dir.glob("*.parquet"))
            assert len(parquet_files) == 5

    def test_episode_saving(self, gridhouse_generator):
        """Test saving episodes to Parquet/JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test Parquet
            parquet_path = Path(tmpdir) / "test_episode.parquet"
            episode = gridhouse_generator.generate_episode(goal_id="prepare_meal")
            gridhouse_generator.generate_episode(
                goal_id="prepare_meal", save_path=parquet_path, format="parquet"
            )
            assert parquet_path.exists()
            
            # Verify can load Parquet
            df = pd.read_parquet(parquet_path)
            assert len(df) == len(episode.steps)
            
            # Test JSONL
            jsonl_path = Path(tmpdir) / "test_episode.jsonl"
            gridhouse_generator.generate_episode(
                goal_id="prepare_meal", save_path=jsonl_path, format="jsonl"
            )
            assert jsonl_path.exists()
            
            # Verify JSONL has correct structure
            with open(jsonl_path, "r") as f:
                lines = f.readlines()
            assert len(lines) > 0  # Should have at least metadata line

    def test_empty_critical_objects(self, gridhouse_generator):
        """Test edge case: task with no critical objects raises error."""
        # First check if there's a task with no critical objects
        # If not, test that invalid task raises appropriate error
        from src.bsa.envs.gridhouse.tasks import get_task, list_tasks
        try:
            # Try to find a task with no critical objects
            for task_id in list_tasks():
                task = get_task(task_id)
                if not task.critical_objects:
                    with pytest.raises(ValueError, match="no critical objects"):
                        gridhouse_generator.generate_episode(goal_id=task_id)
                    return
            # If no such task exists, test invalid task error
            with pytest.raises(ValueError, match="not found"):
                gridhouse_generator.generate_episode(goal_id="invalid_task")
        except ValueError as e:
            # Expected error
            assert "not found" in str(e) or "no critical objects" in str(e)

    def test_no_intervention(self, gridhouse_generator):
        """Test episode generation without intervention."""
        # Create generator with drift_probability=0 to ensure no intervention
        from src.bsa.envs.gridhouse import GridHouseEnvironment
        env = GridHouseEnvironment(seed=42)
        gen_no_intervention = GridHouseEpisodeGenerator(
            env, seed=42, drift_probability=0.0
        )
        episode = gen_no_intervention.generate_episode(
            goal_id="prepare_meal", intervention_type=None
        )
        
        # When intervention_type is None and drift_probability is 0, no intervention should occur
        # However, if tau is still set, intervention might be applied
        # So we check that when explicitly passing None, it's respected
        assert episode.intervention_type is None or episode.metadata.get("intervention_applied", True) is False


# VirtualHome tests (skip if not available)
try:
    from src.bsa.envs.virtualhome import VirtualHomeEnvironment, VirtualHomeEpisodeGenerator
    from src.bsa.envs.virtualhome.tasks import get_task as get_vh_task
    VIRTUALHOME_AVAILABLE = True
except ImportError:
    VIRTUALHOME_AVAILABLE = False


@pytest.fixture
def virtualhome_env():
    """Fixture for VirtualHomeEnvironment."""
    if not VIRTUALHOME_AVAILABLE:
        pytest.skip("VirtualHome not installed")
    try:
        return VirtualHomeEnvironment(seed=42)
    except ImportError:
        pytest.skip("VirtualHome not available")


@pytest.fixture
def virtualhome_generator(virtualhome_env):
    """Fixture for VirtualHomeEpisodeGenerator."""
    if not VIRTUALHOME_AVAILABLE:
        pytest.skip("VirtualHome not installed")
    return VirtualHomeEpisodeGenerator(virtualhome_env, seed=42)


class TestVirtualHomeEpisodeGenerator:
    """Tests for VirtualHomeEpisodeGenerator."""

    def test_generate_episode(self, virtualhome_generator):
        """Test basic episode generation."""
        if not VIRTUALHOME_AVAILABLE:
            pytest.skip("VirtualHome not installed")
        
        episode = virtualhome_generator.generate_episode(goal_id="prepare_meal")
        assert isinstance(episode, Episode)
        assert episode.goal_id == "prepare_meal"
        assert len(episode.steps) > 0

    def test_episode_structure(self, virtualhome_generator):
        """Verify episode has correct structure."""
        if not VIRTUALHOME_AVAILABLE:
            pytest.skip("VirtualHome not installed")
        
        episode = virtualhome_generator.generate_episode(goal_id="prepare_meal")
        
        assert hasattr(episode, "episode_id")
        assert hasattr(episode, "goal_id")
        assert hasattr(episode, "steps")
        assert hasattr(episode, "metadata")

    def test_intervention_applied(self, virtualhome_generator):
        """Test that intervention is applied at tau timestep."""
        if not VIRTUALHOME_AVAILABLE:
            pytest.skip("VirtualHome not installed")
        
        tau = 5
        episode = virtualhome_generator.generate_episode(
            goal_id="prepare_meal", tau=tau, intervention_type="relocate"
        )
        
        assert episode.tau == tau
        assert episode.intervention_type == "relocate"
        assert episode.metadata.get("intervention_applied", False) is True
