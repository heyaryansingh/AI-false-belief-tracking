#!/usr/bin/env python3
"""Fix tables with proper data analysis and add references to paper."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
import pandas as pd
import numpy as np

def main():
    """Fix tables and update paper."""
    print("=" * 70)
    print("Fixing Tables and Adding References")
    print("=" * 70)
    
    # Load raw data
    print("\n[1] Loading and analyzing data...")
    aggregator = AnalysisAggregator()
    raw_df = aggregator.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))
    
    print(f"  Loaded {len(raw_df)} runs")
    
    # Separate by condition for better analysis
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
    
    # Aggregate properly
    print("\n[2] Aggregating data correctly...")
    
    # For summary table: aggregate by model (across all conditions for overall performance)
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    
    # For detection table: use false_belief condition only
    agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    # For intervention table: use false_belief condition only (where interventions matter)
    agg_intervention = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    # Generate improved tables
    print("\n[3] Generating improved tables...")
    
    # Summary table (all conditions)
    table_gen_summary = TableGenerator(agg_by_model)
    summary_md = table_gen_summary.generate_summary_table(format="markdown")
    summary_tex = table_gen_summary.generate_summary_table(format="latex")
    
    # Detection table (false_belief only)
    table_gen_detection = TableGenerator(agg_fb)
    detection_md = table_gen_detection.generate_detection_table(format="markdown")
    detection_tex = table_gen_detection.generate_detection_table(format="latex")
    
    # Task performance (all conditions)
    task_md = table_gen_summary.generate_task_performance_table(format="markdown")
    task_tex = table_gen_summary.generate_task_performance_table(format="latex")
    
    # Intervention table (false_belief only)
    table_gen_intervention = TableGenerator(agg_intervention)
    intervention_md = table_gen_intervention.generate_intervention_table(format="markdown")
    intervention_tex = table_gen_intervention.generate_intervention_table(format="latex")
    
    # Save tables
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    (tables_dir / "summary.md").write_text(summary_md)
    (tables_dir / "summary.tex").write_text(summary_tex)
    (tables_dir / "detection.md").write_text(detection_md)
    (tables_dir / "detection.tex").write_text(detection_tex)
    (tables_dir / "task_performance.md").write_text(task_md)
    (tables_dir / "task_performance.tex").write_text(task_tex)
    (tables_dir / "intervention.md").write_text(intervention_md)
    (tables_dir / "intervention.tex").write_text(intervention_tex)
    
    print("  [OK] Tables saved")
    
    # Print what we found
    print("\n[4] Data summary:")
    print("\nIntervention Quality (False-Belief Condition):")
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_df = fb_df[fb_df["model"] == model]
        prec = model_df["intervention_precision"].dropna()
        rec = model_df["intervention_recall"].dropna()
        over = model_df["over_corrections"].dropna()
        print(f"  {model}:")
        if len(prec) > 0:
            print(f"    Precision: {prec.mean():.3f} ± {prec.std():.3f}")
        if len(rec) > 0:
            print(f"    Recall: {rec.mean():.3f} ± {rec.std():.3f}")
        if len(over) > 0:
            print(f"    Over-corrections: {over.mean():.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
