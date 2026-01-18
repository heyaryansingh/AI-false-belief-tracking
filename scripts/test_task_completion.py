#!/usr/bin/env python3
"""Test task completion detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.envs.gridhouse import GridHouseEpisodeGenerator, GridHouseEnvironment
from bsa.envs.gridhouse.tasks import get_task

# Test task completion detection
env = GridHouseEnvironment(seed=42)
gen = GridHouseEpisodeGenerator(env, seed=42)

print("Testing task completion detection...")
ep = gen.generate_episode(goal_id='find_keys', tau=5, intervention_type='relocate')

print(f"Task completed: {ep.metadata.get('task_completed')}")
print(f"Completion timestep: {ep.metadata.get('completion_timestep')}")
print(f"Num steps: {len(ep.steps)}")

# Check if objects were collected
task = get_task('find_keys')
print(f"\nTask: {task.name}")
print(f"Critical objects: {task.critical_objects}")

# Check final state
final_step = ep.steps[-1]
print(f"\nFinal step objects in locations: {list(final_step.true_object_locations.keys())}")

# Check if keys were picked up
keys_collected = 'keys' not in final_step.true_object_locations
print(f"Keys collected: {keys_collected}")

if ep.metadata.get('task_completed'):
    print("\n[OK] Task completion detection works!")
else:
    print("\n[INFO] Task not completed (may be expected if objects not collected)")
