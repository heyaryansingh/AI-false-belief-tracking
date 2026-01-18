#!/usr/bin/env python3
"""Improve table analysis with better calculations and insights."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
import pandas as pd
import numpy as np

def main():
    """Improve table analysis."""
    print("=" * 70)
    print("Improving Table Analysis")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    aggregator = AnalysisAggregator()
    raw_df = aggregator.load_results(input_path=Path("results/metrics/large_scale_research/results.parquet"))
    
    print(f"  Loaded {len(raw_df)} runs")
    print(f"  Columns: {list(raw_df.columns)}")
    
    # Analyze what we actually have
    print("\n[2] Analyzing available data...")
    
    # Check intervention quality (this is good!)
    print("\nIntervention Quality (False-Belief Condition):")
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
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
    
    # Check detection metrics
    print("\nDetection Metrics (False-Belief Condition):")
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_df = fb_df[fb_df["model"] == model]
        auroc = model_df["false_belief_detection_auroc"].dropna()
        lat = model_df["false_belief_detection_latency"].dropna()
        
        print(f"  {model}:")
        if len(auroc) > 0:
            print(f"    AUROC: {auroc.mean():.3f} ± {auroc.std():.3f} (N={len(auroc)}, unique={len(auroc.unique())})")
        if len(lat) > 0:
            print(f"    Latency: {lat.mean():.2f} ± {lat.std():.2f} (N={len(lat)}, unique={len(lat.unique())})")
    
    # Check task performance
    print("\nTask Performance:")
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_df = raw_df[raw_df["model"] == model]
        completed = model_df["task_completed"].dropna()
        wasted = model_df["num_wasted_actions"].dropna()
        efficiency = model_df["task_efficiency"].dropna()
        
        print(f"  {model}:")
        if len(completed) > 0:
            print(f"    Completed: {completed.sum()}/{len(completed)} ({100*completed.mean():.1f}%)")
        if len(wasted) > 0:
            print(f"    Wasted actions: {wasted.mean():.2f} ± {wasted.std():.2f} (N={len(wasted)})")
        if len(efficiency) > 0:
            print(f"    Efficiency: {efficiency.mean():.3f} ± {efficiency.std():.3f} (N={len(efficiency)})")
    
    # Improve aggregation - focus on false_belief condition for detection metrics
    print("\n[3] Creating improved aggregated data...")
    
    # Aggregate by model only (across all conditions for overall performance)
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    
    # Aggregate by model and condition for condition-specific analysis
    agg_by_model_cond = aggregator.aggregate_metrics(raw_df, group_by=["model", "condition"])
    
    # For detection metrics, use false_belief condition only
    fb_agg = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    print(f"  Aggregated by model: {len(agg_by_model)} groups")
    print(f"  Aggregated by model+condition: {len(agg_by_model_cond)} groups")
    print(f"  False-belief condition: {len(fb_agg)} groups")
    
    # Generate improved tables
    print("\n[4] Generating improved tables...")
    
    # Use false_belief aggregation for detection table
    table_gen_fb = TableGenerator(fb_agg)
    
    # Use overall aggregation for summary table (but improve it)
    table_gen = TableGenerator(agg_by_model)
    
    # Generate tables
    summary_table = table_gen.generate_summary_table(format="markdown")
    detection_table = table_gen_fb.generate_detection_table(format="markdown")
    task_table = table_gen.generate_task_performance_table(format="markdown")
    intervention_table = table_gen_fb.generate_intervention_table(format="markdown")
    
    # Save improved tables
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    (tables_dir / "summary.md").write_text(summary_table)
    (tables_dir / "detection.md").write_text(detection_table)
    (tables_dir / "task_performance.md").write_text(task_table)
    (tables_dir / "intervention.md").write_text(intervention_table)
    
    # Also generate LaTeX versions
    (tables_dir / "summary.tex").write_text(table_gen.generate_summary_table(format="latex"))
    (tables_dir / "detection.tex").write_text(table_gen_fb.generate_detection_table(format="latex"))
    (tables_dir / "task_performance.tex").write_text(table_gen.generate_task_performance_table(format="latex"))
    (tables_dir / "intervention.tex").write_text(table_gen_fb.generate_intervention_table(format="latex"))
    
    print("\n  [OK] Tables generated and saved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
