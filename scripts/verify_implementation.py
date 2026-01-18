#!/usr/bin/env python3
"""Verify all implementation changes are correct."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("Verifying Implementation Changes")
print("=" * 70)

# Test 1: Task completion detection
print("\n[1] Testing task completion detection...")
from bsa.envs.gridhouse import GridHouseEpisodeGenerator, GridHouseEnvironment
from bsa.envs.gridhouse.tasks import get_task

env = GridHouseEnvironment(seed=42)
gen = GridHouseEpisodeGenerator(env, seed=42)

# Generate a test episode
ep = gen.generate_episode(goal_id='find_keys', tau=5, intervention_type='relocate')

# Check metadata
assert 'task_completed' in ep.metadata, "task_completed missing from metadata"
assert 'completion_timestep' in ep.metadata, "completion_timestep missing from metadata"
print("  [OK] Task completion metadata present")

# Test 2: Aggregator boolean handling
print("\n[2] Testing aggregator boolean column handling...")
from bsa.analysis.aggregate import AnalysisAggregator
import pandas as pd
import numpy as np

# Create test data with boolean column
test_df = pd.DataFrame({
    'model': ['test'] * 10,
    'condition': ['test'] * 10,
    'task_completed': [True] * 5 + [False] * 5,
    'num_wasted_actions': [0.0] * 10,
})

agg = AnalysisAggregator()
agg_result = agg.aggregate_metrics(test_df, group_by=['model'])

assert 'task_completed_mean' in agg_result.columns, "task_completed_mean missing"
assert abs(agg_result['task_completed_mean'].iloc[0] - 0.5) < 0.01, "Completion rate incorrect"
print("  [OK] Boolean column aggregation works")

# Test 3: Table generation
print("\n[3] Testing table generation...")
from bsa.analysis.tables import TableGenerator

table_gen = TableGenerator(agg_result)
summary_table = table_gen.generate_summary_table(format="markdown")

assert "Task Completion" in summary_table, "Task Completion missing from summary table"
assert "%" in summary_table or "N/A" in summary_table, "Completion rate not formatted correctly"
print("  [OK] Table generation works")

# Test 4: Evaluator completion timestep
print("\n[4] Testing evaluator...")
from bsa.experiments.evaluator import EpisodeEvaluator
from bsa.agents.helper.reactive import ReactiveHelper

helper = ReactiveHelper()
evaluator = EpisodeEvaluator()

# Create a mock episode with completion
ep.metadata['task_completed'] = True
ep.metadata['completion_timestep'] = 10

metrics = evaluator.evaluate_episode(ep, helper)
assert 'num_steps_to_completion' in metrics, "num_steps_to_completion missing"
print("  [OK] Evaluator works correctly")

print("\n" + "=" * 70)
print("All Implementation Checks Passed!")
print("=" * 70)
