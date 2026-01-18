#!/usr/bin/env python3
"""Monitor regeneration progress."""

import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_progress():
    """Check regeneration progress."""
    print("=" * 70)
    print("Regeneration Progress Monitor")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check episodes
    episodes_dir = Path("data/episodes/large_scale")
    if episodes_dir.exists():
        episode_files = list(episodes_dir.glob("*.parquet"))
        print(f"[1] Episodes: {len(episode_files)} / 10000")
        if len(episode_files) > 0:
            print(f"     Progress: {len(episode_files)/10000*100:.1f}%")
    else:
        print("[1] Episodes: Not started yet")
    
    # Check results
    results_path = Path("results/metrics/large_scale_research/results.parquet")
    if results_path.exists():
        import pandas as pd
        df = pd.read_parquet(results_path)
        print(f"\n[2] Experiment Results: {len(df)} / 450 runs")
        if len(df) > 0:
            print(f"     Progress: {len(df)/450*100:.1f}%")
            
            # Check data quality
            auroc_data = df[df["false_belief_detection_auroc"].notna()]["false_belief_detection_auroc"]
            if len(auroc_data) > 0:
                print(f"\n[3] Data Quality Check:")
                print(f"     AUROC unique values: {len(auroc_data.unique())}")
                print(f"     AUROC std: {auroc_data.std():.4f}")
                if auroc_data.std() > 0.01:
                    print(f"     [OK] Data has variance (real data)")
                else:
                    print(f"     [WARN] Data lacks variance (may be defaults)")
                
                task_completion = df['task_completed'].mean() * 100
                print(f"     Task completion: {task_completion:.2f}%")
    else:
        print("\n[2] Experiment Results: Not started yet")
    
    # Check tables
    tables_dir = Path("results/tables")
    if tables_dir.exists():
        table_files = list(tables_dir.glob("*.md"))
        print(f"\n[4] Tables: {len(table_files)} files")
    else:
        print("\n[4] Tables: Not generated yet")
    
    # Check figures
    figures_dir = Path("results/figures")
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.png"))
        print(f"\n[5] Figures: {len(figure_files)} files")
    else:
        print("\n[5] Figures: Not generated yet")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_progress()
