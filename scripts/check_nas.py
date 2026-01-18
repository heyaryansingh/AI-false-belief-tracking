#!/usr/bin/env python3
"""Check why metrics are N/A."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
import pandas as pd
import numpy as np

agg = AnalysisAggregator()
df = agg.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))

print("=" * 70)
print("N/A Analysis")
print("=" * 70)

print(f"\nTotal runs: {len(df)}")

# Check task_completed
print("\n1. Task Completion:")
print(f"   Non-null values: {df['task_completed'].notna().sum()}")
print(f"   Null values: {df['task_completed'].isna().sum()}")
if df['task_completed'].notna().any():
    print(f"   Unique values: {df['task_completed'].dropna().unique()}")
    print(f"   Mean: {df['task_completed'].mean()}")
else:
    print("   All values are NaN/None")

# Check detection latency
print("\n2. Detection Latency:")
for model in ["reactive", "goal_only", "belief_pf"]:
    model_df = df[df["model"] == model]
    non_null = model_df["false_belief_detection_latency"].notna().sum()
    null = model_df["false_belief_detection_latency"].isna().sum()
    print(f"   {model}:")
    print(f"     Non-null: {non_null}")
    print(f"     Null: {null}")
    if non_null > 0:
        print(f"     Mean: {model_df['false_belief_detection_latency'].mean():.2f}")
        print(f"     Unique values: {model_df['false_belief_detection_latency'].dropna().unique()[:5]}")

# Check steps to completion
print("\n3. Steps to Completion:")
print(f"   Non-null values: {df['num_steps_to_completion'].notna().sum()}")
print(f"   Null values: {df['num_steps_to_completion'].isna().sum()}")
if df['num_steps_to_completion'].notna().any():
    print(f"   Mean: {df['num_steps_to_completion'].mean():.1f}")
    print(f"   Sample values: {df['num_steps_to_completion'].dropna().head(5).tolist()}")

# Check how aggregation handles these
print("\n4. Aggregation Check:")
agg_by_model = agg.aggregate_metrics(df, group_by=["model"])
print("\nAggregated task_completed_mean:")
for model in agg_by_model["group_model"].unique():
    val = agg_by_model[agg_by_model["group_model"] == model]["task_completed_mean"].iloc[0]
    print(f"   {model}: {val}")

print("\nAggregated detection_latency_mean:")
for model in agg_by_model["group_model"].unique():
    val = agg_by_model[agg_by_model["group_model"] == model]["false_belief_detection_latency_mean"].iloc[0]
    print(f"   {model}: {val}")
