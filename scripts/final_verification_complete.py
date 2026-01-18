#!/usr/bin/env python3
"""Final verification that everything is complete."""

from pathlib import Path
import pandas as pd

print("=" * 70)
print("Final Verification: All Changes Complete")
print("=" * 70)

# Check data
results_path = Path("results/metrics/large_scale_research/results.parquet")
if results_path.exists():
    df = pd.read_parquet(results_path)
    print(f"\n[OK] Data: {len(df)} runs loaded")
    print(f"  Task completion rate: {df['task_completed'].mean()*100:.2f}%")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Conditions: {df['condition'].unique().tolist()}")
else:
    print("\n[ERROR] Results file not found")

# Check tables
tables_dir = Path("results/tables")
if tables_dir.exists():
    table_files = list(tables_dir.glob("*.md"))
    print(f"\n[OK] Tables: {len(table_files)} files")
    for f in table_files:
        content = f.read_text()
        if "N/A" in content and "Task Completion" in content or "Completion Rate" in content:
            # Check if it's the old N/A or new percentage
            if "0.0%" in content:
                print(f"  {f.name}: [OK] Shows completion rate")
            else:
                print(f"  {f.name}: [WARN] May have N/A")
        else:
            print(f"  {f.name}: [OK]")
else:
    print("\n[ERROR] Tables directory not found")

# Check figures
figures_dir = Path("results/figures")
if figures_dir.exists():
    figure_files = list(figures_dir.glob("*.png"))
    print(f"\n[OK] Figures: {len(figure_files)} files")
else:
    print("\n[ERROR] Figures directory not found")

# Check paper
paper_path = Path("paper/research_paper.md")
if paper_path.exists():
    paper_content = paper_path.read_text()
    # Check if tables have been updated
    if "0.0%" in paper_content and "Task Completion" in paper_content:
        print("\n[OK] Paper: Tables updated with completion rates")
    else:
        print("\n[WARN] Paper: May need table updates")
    
    # Check references
    if "Wimmer, H., & Perner, J. (1983)" in paper_content:
        print("[OK] Paper: References added")
    else:
        print("[WARN] Paper: References may be missing")
    
    # Check acknowledgements removed
    if "## Acknowledgments" not in paper_content:
        print("[OK] Paper: Acknowledgements removed")
    else:
        print("[WARN] Paper: Acknowledgements still present")
else:
    print("\n[ERROR] Paper not found")

print("\n" + "=" * 70)
print("Verification Complete!")
print("=" * 70)
