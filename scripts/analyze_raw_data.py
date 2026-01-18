#!/usr/bin/env python3
"""Analyze raw data to understand what's actually happening."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
import pandas as pd
import numpy as np

agg = AnalysisAggregator()
raw_df = agg.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))

print("=" * 70)
print("Raw Data Analysis")
print("=" * 70)

# Check intervention quality by condition
print("\nIntervention Quality by Condition:")
for condition in ["control", "false_belief", "seen_relocation"]:
    cond_df = raw_df[raw_df["condition"] == condition]
    print(f"\n{condition.upper()}:")
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_df = cond_df[cond_df["model"] == model]
        prec = model_df["intervention_precision"].dropna()
        rec = model_df["intervention_recall"].dropna()
        over = model_df["over_corrections"].dropna()
        
        if len(prec) > 0 or len(rec) > 0:
            print(f"  {model}:")
            if len(prec) > 0:
                print(f"    Precision: {prec.mean():.3f} ± {prec.std():.3f} (N={len(prec)})")
            if len(rec) > 0:
                print(f"    Recall: {rec.mean():.3f} ± {rec.std():.3f} (N={len(rec)})")
            if len(over) > 0:
                print(f"    Over-corrections: {over.mean():.1f}% (N={len(over)})")

# Check what the paper claims vs reality
print("\n" + "=" * 70)
print("Paper Claims vs Actual Data (False-Belief Condition)")
print("=" * 70)
fb_df = raw_df[raw_df["condition"] == "false_belief"]

print("\nPaper Claims:")
print("  belief_pf: Precision 0.291, Recall 0.400, Over-corrections 34.9%")
print("  goal_only: Precision 0.193, Recall 0.260, Over-corrections 40.3%")
print("  reactive: Precision 0.172, Recall 0.240, Over-corrections 41.4%")

print("\nActual Data (False-Belief Condition):")
for model in ["reactive", "goal_only", "belief_pf"]:
    model_df = fb_df[fb_df["model"] == model]
    prec = model_df["intervention_precision"].dropna()
    rec = model_df["intervention_recall"].dropna()
    over = model_df["over_corrections"].dropna()
    
    print(f"  {model}:")
    if len(prec) > 0:
        print(f"    Precision: {prec.mean():.3f} ± {prec.std():.3f} (N={len(prec)})")
    if len(rec) > 0:
        print(f"    Recall: {rec.mean():.3f} ± {rec.std():.3f} (N={len(rec)})")
    if len(over) > 0:
        print(f"    Over-corrections: {over.mean():.1f}% (N={len(over)})")

# Check if there's a calculation issue
print("\n" + "=" * 70)
print("Checking for Calculation Issues")
print("=" * 70)

# Maybe the paper was using a different aggregation?
# Let's check if aggregating differently gives different results
print("\nAggregating by model only (all conditions):")
agg_all = agg.aggregate_metrics(raw_df, group_by=["model"])
for _, row in agg_all.iterrows():
    model = row["group_model"]
    prec_mean = row.get("intervention_precision_mean", np.nan)
    rec_mean = row.get("intervention_recall_mean", np.nan)
    over_mean = row.get("over_corrections_mean", np.nan)
    print(f"  {model}:")
    if not np.isnan(prec_mean):
        print(f"    Precision: {prec_mean:.3f}")
    if not np.isnan(rec_mean):
        print(f"    Recall: {rec_mean:.3f}")
    if not np.isnan(over_mean):
        print(f"    Over-corrections: {over_mean:.1f}%")
