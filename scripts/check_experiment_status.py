#!/usr/bin/env python3
"""Check experiment status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
import pandas as pd

config = load_config(Path("configs/experiments/exp_large_scale.yaml"))
exp_config = config.get("experiment", {})

expected_runs = exp_config.get("num_runs", 50) * len(exp_config.get("models", [])) * len(exp_config.get("conditions", []))

results_path = Path("results/metrics/large_scale_research/results.parquet")

if results_path.exists():
    df = pd.read_parquet(results_path)
    current_runs = len(df)
    progress = (current_runs / expected_runs * 100) if expected_runs > 0 else 0
    
    print(f"Expected runs: {expected_runs}")
    print(f"Current runs: {current_runs}")
    print(f"Progress: {progress:.1f}%")
    
    if current_runs >= expected_runs:
        print("\n[OK] Experiments complete! Ready for analysis.")
    else:
        print(f"\n[INFO] Experiments still running... ({expected_runs - current_runs} runs remaining)")
else:
    print("No results file found. Experiments may still be running or not started.")
