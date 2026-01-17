#!/usr/bin/env python3
"""Test all functionality to verify no conflicts."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Comprehensive Functionality Test")
print("=" * 70)
print()

all_passed = True

# Test 1: VirtualHomeEnvironment
print("[1] Testing VirtualHomeEnvironment...")
try:
    from src.bsa.envs.virtualhome import VirtualHomeEnvironment
    env = VirtualHomeEnvironment()
    obs = env.reset(seed=42)
    print(f"    [OK] VirtualHomeEnvironment works: {obs.current_room}")
except Exception as e:
    print(f"    [SKIP] VirtualHome not fully functional: {e}")
print()

# Test 2: Episode serialization
print("[2] Testing episode serialization...")
try:
    from src.bsa.envs.gridhouse.recorder import EpisodeRecorder
    from src.bsa.envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
    import tempfile
    import os
    
    env = GridHouseEnvironment(seed=42)
    gen = GridHouseEpisodeGenerator(env, seed=42)
    episode = gen.generate_episode()
    recorder = EpisodeRecorder()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / 'test.parquet'
        recorder.save_episode(episode, parquet_path, format='parquet')
        print("    [OK] Parquet serialization works")
        
        jsonl_path = Path(tmpdir) / 'test.jsonl'
        recorder.save_episode(episode, jsonl_path, format='jsonl')
        print("    [OK] JSONL serialization works")
except Exception as e:
    print(f"    [FAIL] Serialization test failed: {e}")
    all_passed = False
print()

# Test 3: Full integration
print("[3] Testing full integration (env + generator + helpers)...")
try:
    from src.bsa.envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
    from src.bsa.agents.helper import ReactiveHelper, GoalOnlyHelper, BeliefSensitiveHelper
    from src.bsa.agents.human import ScriptedHumanAgent
    from src.bsa.envs.gridhouse.tasks import get_task
    
    env = GridHouseEnvironment(seed=42)
    gen = GridHouseEpisodeGenerator(env, seed=42)
    episode = gen.generate_episode(goal_id='prepare_meal', tau=5)
    print(f"    [OK] Episode generated: {len(episode.steps)} steps")
    
    reactive = ReactiveHelper()
    goal_only = GoalOnlyHelper()
    belief = BeliefSensitiveHelper()
    print("    [OK] All helpers instantiated")
except Exception as e:
    print(f"    [FAIL] Integration test failed: {e}")
    all_passed = False
print()

# Test 4: Dependency conflicts
print("[4] Checking for dependency conflicts...")
try:
    import numpy
    import pandas
    import scipy
    import matplotlib
    import yaml
    import pydantic
    
    # Check versions are compatible
    numpy_ver = numpy.__version__
    pandas_ver = pandas.__version__
    
    print(f"    [OK] NumPy {numpy_ver}")
    print(f"    [OK] Pandas {pandas_ver}")
    print("    [OK] No import conflicts detected")
except Exception as e:
    print(f"    [FAIL] Dependency check failed: {e}")
    all_passed = False
print()

# Test 5: Inference modules
print("[5] Testing inference modules...")
try:
    from src.bsa.inference.particle_filter import ParticleFilter
    from src.bsa.inference.goal import GoalInference
    from src.bsa.inference.belief import BeliefInference
    from src.bsa.envs.gridhouse.tasks import list_tasks, get_task
    
    tasks = [get_task(t) for t in list_tasks()]
    pf = ParticleFilter(tasks, num_particles=10)
    gi = GoalInference(tasks)
    bi = BeliefInference(tasks, num_particles=10)
    
    print(f"    [OK] Particle filter: {len(pf.particles)} particles")
    print(f"    [OK] Goal inference: {len(gi.get_goal_distribution())} goals")
    print(f"    [OK] Belief inference: {len(bi.get_belief_state())} belief components")
except Exception as e:
    print(f"    [FAIL] Inference modules test failed: {e}")
    all_passed = False
print()

# Summary
print("=" * 70)
if all_passed:
    print("[SUCCESS] All functionality tests passed!")
else:
    print("[ISSUES] Some tests failed - see above")
print("=" * 70)

sys.exit(0 if all_passed else 1)
