"""Tests for environment interface compliance."""

import pytest
from src.bsa.envs.base import Environment
from src.bsa.envs.gridhouse import GridHouseEnvironment
from src.bsa.common.types import Action


def test_gridhouse_implements_interface():
    """Test that GridHouse implements Environment interface."""
    env = GridHouseEnvironment(seed=42)
    assert isinstance(env, Environment)


def test_gridhouse_reset():
    """Test environment reset."""
    env = GridHouseEnvironment(seed=42)
    obs = env.reset(seed=42)
    assert obs is not None
    assert obs.agent_id == "human"


def test_gridhouse_step():
    """Test environment step."""
    env = GridHouseEnvironment(seed=42)
    env.reset(seed=42)
    obs, reward, done, info = env.step(Action.MOVE, "human")
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_gridhouse_get_true_state():
    """Test getting true state."""
    env = GridHouseEnvironment(seed=42)
    env.reset(seed=42)
    state = env.get_true_state()
    assert "object_locations" in state
    assert "agent_positions" in state
    assert "timestep" in state


def test_gridhouse_get_visible_state():
    """Test getting visible state."""
    env = GridHouseEnvironment(seed=42)
    env.reset(seed=42)
    obs = env.get_visible_state("human")
    assert obs.agent_id == "human"
    assert isinstance(obs.visible_objects, list)
    assert isinstance(obs.visible_containers, list)
