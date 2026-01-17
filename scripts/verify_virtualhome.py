#!/usr/bin/env python3
"""End-to-end verification script for VirtualHome integration."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("VirtualHome End-to-End Verification")
print("=" * 70)
print()

all_passed = True
issues = []

# Check 1: VirtualHome installation
print("[1] Checking VirtualHome installation...")
try:
    import virtualhome
    print(f"    [OK] VirtualHome installed")
except ImportError:
    print("    [SKIP] VirtualHome not installed - using GridHouse fallback")
    print("    Note: VirtualHome is optional. GridHouse fallback works.")
    sys.exit(0)

# Check 2: VirtualHomeEnvironment import
print("[2] Checking VirtualHomeEnvironment import...")
try:
    from src.bsa.envs.virtualhome import VirtualHomeEnvironment
    print("    [OK] VirtualHomeEnvironment imports successfully")
except ImportError as e:
    print(f"    [FAIL] Import error: {e}")
    all_passed = False
    issues.append(f"VirtualHomeEnvironment import: {e}")
print()

# Check 3: Environment creation and reset
print("[3] Testing VirtualHomeEnvironment creation and reset...")
try:
    env = VirtualHomeEnvironment(seed=42)
    obs = env.reset(seed=42)
    print(f"    [OK] Environment created and reset successfully")
    print(f"    [OK] Initial observation: room={obs.current_room}")
except Exception as e:
    print(f"    [FAIL] Environment error: {e}")
    all_passed = False
    issues.append(f"VirtualHomeEnvironment reset: {e}")
print()

# Check 4: Episode generation
print("[4] Testing episode generation...")
try:
    from src.bsa.envs.virtualhome import VirtualHomeEpisodeGenerator
    generator = VirtualHomeEpisodeGenerator(env, seed=42)
    episode = generator.generate_episode(goal_id="prepare_meal", tau=5)
    print(f"    [OK] Episode generated: {len(episode.steps)} steps")
    print(f"    [OK] Goal: {episode.goal_id}, Tau: {episode.tau}")
except Exception as e:
    print(f"    [FAIL] Episode generation error: {e}")
    all_passed = False
    issues.append(f"Episode generation: {e}")
print()

# Check 5: Episode structure
print("[5] Verifying episode structure...")
try:
    assert hasattr(episode, "episode_id")
    assert hasattr(episode, "goal_id")
    assert hasattr(episode, "steps")
    assert hasattr(episode, "metadata")
    assert len(episode.steps) > 0
    print("    [OK] Episode structure is correct")
except AssertionError as e:
    print(f"    [FAIL] Episode structure error: {e}")
    all_passed = False
    issues.append(f"Episode structure: {e}")
print()

# Check 6: Helper agents integration
print("[6] Testing helper agents with VirtualHome...")
try:
    from src.bsa.agents.helper import ReactiveHelper, GoalOnlyHelper, BeliefSensitiveHelper
    
    helper_obs = env.get_visible_state("helper")
    
    reactive = ReactiveHelper()
    action1 = reactive.plan_action(helper_obs)
    print("    [OK] ReactiveHelper works")
    
    goal_only = GoalOnlyHelper()
    action2 = goal_only.plan_action(helper_obs)
    print("    [OK] GoalOnlyHelper works")
    
    belief = BeliefSensitiveHelper(num_particles=10)
    action3 = belief.plan_action(helper_obs)
    print("    [OK] BeliefSensitiveHelper works")
except Exception as e:
    print(f"    [FAIL] Helper agents error: {e}")
    all_passed = False
    issues.append(f"Helper agents: {e}")
print()

# Check 7: Episode serialization
print("[7] Testing episode serialization...")
try:
    from src.bsa.envs.virtualhome import VirtualHomeEpisodeRecorder
    import tempfile
    
    recorder = VirtualHomeEpisodeRecorder()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "test_episode.parquet"
        recorder.save_episode(episode, parquet_path, format="parquet")
        assert parquet_path.exists()
        print("    [OK] Parquet serialization works")
        
        jsonl_path = Path(tmpdir) / "test_episode.jsonl"
        recorder.save_episode(episode, jsonl_path, format="jsonl")
        assert jsonl_path.exists()
        print("    [OK] JSONL serialization works")
except Exception as e:
    print(f"    [FAIL] Serialization error: {e}")
    all_passed = False
    issues.append(f"Episode serialization: {e}")
print()

# Check 8: Observability module
print("[8] Testing observability module...")
try:
    from src.bsa.envs.virtualhome import get_scene_state, get_agent_view
    
    scene_state = get_scene_state(env)
    assert isinstance(scene_state, dict)
    print("    [OK] get_scene_state works")
    
    agent_view = get_agent_view(env, "human")
    assert isinstance(agent_view, dict)
    print("    [OK] get_agent_view works")
except Exception as e:
    print(f"    [FAIL] Observability error: {e}")
    all_passed = False
    issues.append(f"Observability: {e}")
print()

# Summary
print("=" * 70)
if all_passed:
    print("[SUCCESS] All VirtualHome integration checks passed!")
    print("VirtualHome is fully integrated and working.")
else:
    print("[ISSUES] Some checks failed:")
    for issue in issues:
        print(f"  - {issue}")
print("=" * 70)

sys.exit(0 if all_passed else 1)
