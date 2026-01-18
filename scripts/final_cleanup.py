#!/usr/bin/env python3
"""Final cleanup of duplicate analysis directories."""

from pathlib import Path
import shutil

print("=" * 70)
print("Final Cleanup")
print("=" * 70)

# Remove duplicate analysis directories
print("\n[1] Removing duplicate analysis directories...")

# Remove results/analysis/figures if empty or duplicate
analysis_figures = Path("results/analysis/figures")
if analysis_figures.exists():
    figs = list(analysis_figures.glob("*.png"))
    if len(figs) == 0:
        analysis_figures.rmdir()
        print("  Removed: results/analysis/figures/ (empty)")
    else:
        print(f"  Warning: results/analysis/figures/ has {len(figs)} files")

# Keep results/analysis/large_scale/ but remove empty subdirs
large_scale = Path("results/analysis/large_scale")
if large_scale.exists():
    large_scale_figures = large_scale / "figures"
    if large_scale_figures.exists():
        figs = list(large_scale_figures.glob("*.png"))
        if len(figs) == 0:
            large_scale_figures.rmdir()
            print("  Removed: results/analysis/large_scale/figures/ (empty)")

print("\n[2] Verifying data loading...")
# Test loading results from the correct path
try:
    import sys
    sys.path.insert(0, "src")
    from pathlib import Path
    import pandas as pd
    
    results_file = Path("results/metrics/large_scale_research/results.parquet")
    if results_file.exists():
        df = pd.read_parquet(results_file)
        print(f"  [OK] Results file loads: {len(df)} rows")
    else:
        print(f"  [FAIL] Results file not found")
except Exception as e:
    print(f"  [FAIL] Error loading results: {e}")

print("\n" + "=" * 70)
print("Cleanup Complete")
print("=" * 70)
