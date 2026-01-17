#!/usr/bin/env python3
"""Comprehensive verification script for project setup.

Verifies:
- Virtual environment setup
- All dependencies install correctly
- No dependency conflicts
- All modules import successfully
- Core functionality works
"""

import sys
import os
import importlib
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major != 3:
        return False, f"Python {version_str} - Python 3 required"
    
    if version.minor < 9:
        return False, f"Python {version_str} - Python 3.9+ required"
    
    if version.minor >= 12:
        return False, f"Python {version_str} - Python 3.9-3.11 recommended for VirtualHome"
    
    return True, f"Python {version_str}"


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported.
    
    Args:
        module_name: Module to import (e.g., 'numpy')
        package_name: Display name (if different from module_name)
    
    Returns:
        (success, message)
    """
    if package_name is None:
        package_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"{package_name} {version}"
    except ImportError as e:
        return False, f"{package_name} - ImportError: {e}"


def check_numpy_version() -> Tuple[bool, str]:
    """Check NumPy version is compatible (<2.0 for VirtualHome)."""
    try:
        import numpy as np
        version = np.__version__
        major_version = int(version.split('.')[0])
        
        if major_version >= 2:
            return False, f"NumPy {version} - Version >=2.0 may conflict with VirtualHome (need <2.0)"
        return True, f"NumPy {version} (compatible)"
    except ImportError:
        return False, "NumPy not installed"


def check_project_modules() -> List[Tuple[bool, str]]:
    """Check all project modules can be imported."""
    results = []
    
    # Core modules
    modules = [
        ("src.bsa.common.types", "Common types"),
        ("src.bsa.common.seeding", "Seeding utilities"),
        ("src.bsa.envs.base", "Environment base"),
        ("src.bsa.envs.gridhouse.env", "GridHouse environment"),
        ("src.bsa.envs.gridhouse.tasks", "GridHouse tasks"),
        ("src.bsa.envs.gridhouse.episode_generator", "GridHouse episode generator"),
        ("src.bsa.envs.gridhouse.recorder", "Episode recorder"),
        ("src.bsa.agents.human.scripted_human", "Scripted human agent"),
        ("src.bsa.agents.helper.base", "Helper agent base"),
        ("src.bsa.agents.helper.reactive", "Reactive helper"),
        ("src.bsa.agents.helper.goal_only", "Goal-only helper"),
        ("src.bsa.agents.helper.belief_sensitive", "Belief-sensitive helper"),
        ("src.bsa.inference.goal", "Goal inference"),
        ("src.bsa.inference.particle_filter", "Particle filter"),
        ("src.bsa.inference.belief", "Belief inference"),
        ("src.bsa.inference.likelihood", "Likelihood models"),
    ]
    
    for module_name, display_name in modules:
        success, msg = check_import(module_name, display_name)
        results.append((success, msg))
    
    return results


def check_virtualhome() -> Tuple[bool, str]:
    """Check VirtualHome availability."""
    try:
        from src.bsa.envs.virtualhome import VirtualHomeEnvironment
        return True, "VirtualHomeEnvironment available"
    except ImportError as e:
        if "VirtualHome is not installed" in str(e):
            return False, "VirtualHome not installed (GridHouse fallback available)"
        return False, f"VirtualHome import error: {e}"


def test_gridhouse_functionality() -> Tuple[bool, str]:
    """Test GridHouse environment functionality."""
    try:
        from src.bsa.envs.gridhouse import GridHouseEnvironment
        from src.bsa.common.types import Action
        
        env = GridHouseEnvironment(seed=42)
        obs = env.reset(seed=42)
        
        if obs is None:
            return False, "GridHouse reset returned None"
        
        if not hasattr(obs, 'current_room'):
            return False, "Observation missing current_room attribute"
        
        # Test step
        obs2, reward, done, info = env.step(Action.MOVE, "human")
        
        # Test get_object_locations
        locations = env.get_object_locations()
        
        return True, f"GridHouse functional (reset, step, get_object_locations work)"
    except Exception as e:
        return False, f"GridHouse test failed: {e}"


def test_helper_agents() -> Tuple[bool, str]:
    """Test helper agents functionality."""
    try:
        from src.bsa.agents.helper import ReactiveHelper, GoalOnlyHelper, BeliefSensitiveHelper
        from src.bsa.common.types import Observation
        
        obs = Observation("helper", [], [], "kitchen", (0, 0, 0), 0)
        
        # Test reactive helper
        reactive = ReactiveHelper()
        action1 = reactive.plan_action(obs)
        state1 = reactive.get_belief_state()
        
        # Test goal-only helper
        goal_only = GoalOnlyHelper()
        action2 = goal_only.plan_action(obs)
        state2 = goal_only.get_belief_state()
        
        # Test belief-sensitive helper
        belief_sensitive = BeliefSensitiveHelper(num_particles=10)
        action3 = belief_sensitive.plan_action(obs)
        state3 = belief_sensitive.get_belief_state()
        
        return True, "All helper agents functional"
    except Exception as e:
        return False, f"Helper agents test failed: {e}"


def test_episode_generation() -> Tuple[bool, str]:
    """Test episode generation."""
    try:
        from src.bsa.envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
        
        env = GridHouseEnvironment(seed=42)
        generator = GridHouseEpisodeGenerator(env, seed=42)
        
        episode = generator.generate_episode(goal_id="prepare_meal", tau=5)
        
        if episode is None:
            return False, "Episode generation returned None"
        
        if not hasattr(episode, 'steps'):
            return False, "Episode missing steps attribute"
        
        if len(episode.steps) == 0:
            return False, "Episode has no steps"
        
        return True, f"Episode generation works ({len(episode.steps)} steps)"
    except Exception as e:
        return False, f"Episode generation test failed: {e}"


def main():
    """Run comprehensive verification."""
    print("=" * 70)
    print("Comprehensive Setup Verification")
    print("=" * 70)
    print()
    
    all_passed = True
    issues = []
    
    # Check Python version
    print("[1] Checking Python version...")
    success, msg = check_python_version()
    print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
    if not success:
        all_passed = False
        issues.append(f"Python version: {msg}")
    print()
    
    # Check core dependencies
    print("[2] Checking core dependencies...")
    core_deps = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("pydantic", "Pydantic"),
        ("tqdm", "tqdm"),
        ("pyarrow", "PyArrow"),
        ("hydra", "Hydra"),
    ]
    
    for module, name in core_deps:
        success, msg = check_import(module, name)
        print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
        if not success:
            all_passed = False
            issues.append(f"{name}: {msg}")
    print()
    
    # Check NumPy version specifically
    print("[3] Checking NumPy version compatibility...")
    success, msg = check_numpy_version()
    print(f"    {'[OK]' if success else '[WARN]'} {msg}")
    if not success:
        issues.append(f"NumPy version: {msg}")
    print()
    
    # Check project modules
    print("[4] Checking project modules...")
    module_results = check_project_modules()
    for success, msg in module_results:
        print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
        if not success:
            all_passed = False
            issues.append(f"Module: {msg}")
    print()
    
    # Check VirtualHome
    print("[5] Checking VirtualHome...")
    success, msg = check_virtualhome()
    status = "[OK]" if success else "[SKIP]" if "not installed" in msg else "[FAIL]"
    print(f"    {status} {msg}")
    if not success and "FAIL" in status:
        all_passed = False
        issues.append(f"VirtualHome: {msg}")
    print()
    
    # Test GridHouse functionality
    print("[6] Testing GridHouse functionality...")
    success, msg = test_gridhouse_functionality()
    print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
    if not success:
        all_passed = False
        issues.append(f"GridHouse: {msg}")
    print()
    
    # Test helper agents
    print("[7] Testing helper agents...")
    success, msg = test_helper_agents()
    print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
    if not success:
        all_passed = False
        issues.append(f"Helper agents: {msg}")
    print()
    
    # Test episode generation
    print("[8] Testing episode generation...")
    success, msg = test_episode_generation()
    print(f"    {'[OK]' if success else '[FAIL]'} {msg}")
    if not success:
        all_passed = False
        issues.append(f"Episode generation: {msg}")
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print("[SUCCESS] All checks passed!")
    else:
        print("[ISSUES] Some checks failed:")
        for issue in issues:
            print(f"  - {issue}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
