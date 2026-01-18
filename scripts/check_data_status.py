#!/usr/bin/env python3
"""Check data status and quality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

results_path = Path("results/metrics/large_scale_research/results.parquet")

if results_path.exists():
    df = pd.read_parquet(results_path)
    print(f"Results file exists: {len(df)} runs")
    print(f"\nTask completion: {df['task_completed'].mean()*100:.2f}%")
    
    auroc_data = df[df["false_belief_detection_auroc"].notna()]["false_belief_detection_auroc"]
    if len(auroc_data) > 0:
        print(f"\nAUROC Statistics:")
        print(f"  Mean: {auroc_data.mean():.3f}")
        print(f"  Std: {auroc_data.std():.3f}")
        print(f"  Min: {auroc_data.min():.3f}")
        print(f"  Max: {auroc_data.max():.3f}")
        print(f"  Unique values: {len(auroc_data.unique())}")
        print(f"  Sample values: {auroc_data.head(10).tolist()}")
    
    print(f"\nModels: {df['model'].unique().tolist()}")
    print(f"Conditions: {df['condition'].unique().tolist()}")
    
    # Check if data looks real or test data
    if auroc_data.std() == 0.0 and len(auroc_data.unique()) <= 3:
        print("\n[WARNING] Data appears to be test/dummy data (no variance)")
    else:
        print("\n[OK] Data appears to have real variance")
else:
    print("Results file does not exist yet")
