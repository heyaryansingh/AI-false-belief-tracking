#!/usr/bin/env python3
"""Complete regeneration of all analysis, tables, figures, and paper."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
from bsa.viz.plots import PlotGenerator
from bsa.analysis.report import ReportGenerator
from bsa.common.config import load_config
import pandas as pd

def main():
    """Regenerate all analysis outputs."""
    print("=" * 70)
    print("Complete Analysis Regeneration")
    print("=" * 70)
    
    # Load comprehensive analysis config
    config_path = Path("configs/analysis/comprehensive.yaml")
    if not config_path.exists():
        print(f"Warning: Comprehensive config not found: {config_path}")
        print("Using default settings...")
        config = {}
    else:
        config = load_config(config_path)
    
    # Load results
    print("\n[1] Loading results...")
    aggregator = AnalysisAggregator()
    results_path = Path("results/metrics/large_scale_research/results.parquet")
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Please run experiments first.")
        return 1
    
    raw_df = aggregator.load_results(input_path=results_path)
    print(f"  Loaded {len(raw_df)} runs")
    
    # Aggregate data
    print("\n[2] Aggregating data...")
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    agg_by_model_cond = aggregator.aggregate_metrics(raw_df, group_by=["model", "condition"])
    
    # False-belief condition only
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
    agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    print(f"  Aggregated by model: {len(agg_by_model)} groups")
    print(f"  Aggregated by model+condition: {len(agg_by_model_cond)} groups")
    print(f"  False-belief condition: {len(agg_fb)} groups")
    
    # Generate tables
    print("\n[3] Generating tables...")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary table (all conditions)
    table_gen_summary = TableGenerator(agg_by_model)
    summary_md = table_gen_summary.generate_summary_table(format="markdown")
    summary_tex = table_gen_summary.generate_summary_table(format="latex")
    (tables_dir / "summary.md").write_text(summary_md)
    (tables_dir / "summary.tex").write_text(summary_tex)
    print("  [OK] Summary table")
    
    # Detection table (false-belief only)
    table_gen_detection = TableGenerator(agg_fb)
    detection_md = table_gen_detection.generate_detection_table(format="markdown")
    detection_tex = table_gen_detection.generate_detection_table(format="latex")
    (tables_dir / "detection.md").write_text(detection_md)
    (tables_dir / "detection.tex").write_text(detection_tex)
    print("  [OK] Detection table")
    
    # Task performance table (all conditions)
    task_md = table_gen_summary.generate_task_performance_table(format="markdown")
    task_tex = table_gen_summary.generate_task_performance_table(format="latex")
    (tables_dir / "task_performance.md").write_text(task_md)
    (tables_dir / "task_performance.tex").write_text(task_tex)
    print("  [OK] Task performance table")
    
    # Intervention table (false-belief only)
    intervention_md = table_gen_detection.generate_intervention_table(format="markdown")
    intervention_tex = table_gen_detection.generate_intervention_table(format="latex")
    (tables_dir / "intervention.md").write_text(intervention_md)
    (tables_dir / "intervention.tex").write_text(intervention_tex)
    print("  [OK] Intervention table")
    
    # Generate figures
    print("\n[4] Generating figures...")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_gen = PlotGenerator(agg_by_model_cond, raw_df=raw_df, output_dir=figures_dir)
    
    # Generate all comprehensive plots
    plot_methods = [
        ("plot_detection_auroc", "detection_auroc.png"),
        ("plot_detection_auroc_detailed", "detection_auroc_detailed.png"),
        ("plot_detection_auroc_by_condition", "detection_auroc_by_condition.png"),
        ("plot_detection_latency_boxplot", "detection_latency_boxplot.png"),
        ("plot_detection_latency_cdf", "detection_latency_cdf.png"),
        ("plot_detection_latency_histogram", "detection_latency_histogram.png"),
        ("plot_task_performance", "task_performance.png"),
        ("plot_task_performance_detailed", "task_performance_detailed.png"),
        ("plot_intervention_quality", "intervention_quality.png"),
        ("plot_intervention_pr_scatter", "intervention_pr_scatter.png"),
        ("plot_intervention_timing_dist", "intervention_timing_dist.png"),
        ("plot_belief_timeline", "belief_timeline.png"),
        ("plot_goal_inference_by_condition", "goal_inference_by_condition.png"),
        ("plot_condition_comparison_heatmap", "condition_comparison_heatmap.png"),
        ("plot_model_comparison_heatmap", "model_comparison_heatmap.png"),
        ("plot_tau_effect", "tau_effect.png"),
        ("plot_summary_figure", "summary_figure.png"),
    ]
    
    for method_name, filename in plot_methods:
        try:
            method = getattr(plot_gen, method_name)
            method(save_path=figures_dir / filename)
            print(f"  [OK] {filename}")
        except Exception as e:
            print(f"  [WARN] {filename}: {e}")
    
    # Generate report
    print("\n[5] Generating report...")
    report_dir = Path("results/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(
        aggregated_df=agg_by_model_cond,
        output_dir=report_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    
    report_path = report_gen.generate_report()
    print(f"  [OK] Report: {report_path}")
    
    print("\n" + "=" * 70)
    print("Regeneration Complete!")
    print("=" * 70)
    print(f"\nTables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Report: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
