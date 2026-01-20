#!/usr/bin/env python3
"""Generate professional publication-quality figures from Phase 9 validation data.

This script generates all visualizations needed for the research paper
using the fixed experimental data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.viz.plots import PlotGenerator, generate_all_plots


def main():
    print("=" * 70)
    print("Phase 9: Generate Publication-Quality Figures")
    print("=" * 70)

    # Load Phase 9 validation data
    data_path = Path("results/metrics/phase9_validation/results.parquet")
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run 'python scripts/run_phase9_experiments.py' first.")
        return 1

    print(f"\nLoading data from: {data_path}")
    raw_df = pd.read_parquet(data_path)
    print(f"  Loaded {len(raw_df)} rows")

    # Show data summary
    print("\n** Data Summary **")
    print(f"  Models: {raw_df['model'].unique().tolist()}")
    print(f"  Conditions: {raw_df['condition'].unique().tolist()}")
    print(f"  Runs: {raw_df['run'].nunique()}")

    # Create aggregated DataFrame for some plots
    agg_df = raw_df.groupby(["model", "condition"]).agg({
        "auroc": ["mean", "std", "count"],
        "latency": ["mean", "std"],
        "fpr": ["mean", "std"],
        "task_completed": ["mean", "std"],
        "efficiency": ["mean", "std"],
        "wasted_actions": ["mean", "std"],
        "interventions": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
    }).reset_index()

    # Flatten column names
    agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns.values]

    # Rename for compatibility with PlotGenerator
    rename_map = {
        "model": "group_model",
        "condition": "group_condition",
        "auroc_mean": "false_belief_detection_auroc_mean",
        "auroc_std": "false_belief_detection_auroc_std",
        "latency_mean": "false_belief_detection_latency_mean",
        "latency_std": "false_belief_detection_latency_std",
        "fpr_mean": "false_belief_detection_fpr_mean",
        "task_completed_mean": "task_completed_mean",
        "efficiency_mean": "task_efficiency_mean",
        "wasted_actions_mean": "num_wasted_actions_mean",
        "interventions_mean": "num_interventions_mean",
        "precision_mean": "intervention_precision_mean",
        "recall_mean": "intervention_recall_mean",
        "over_corrections_mean": "over_corrections_mean",
    }
    for old, new in rename_map.items():
        if old in agg_df.columns:
            agg_df = agg_df.rename(columns={old: new})

    # Rename raw_df columns for compatibility
    raw_rename = {
        "auroc": "false_belief_detection_auroc",
        "latency": "false_belief_detection_latency",
        "fpr": "false_belief_detection_fpr",
        "task_completed": "task_completed",
        "efficiency": "task_efficiency",
        "wasted_actions": "num_wasted_actions",
        "interventions": "num_interventions",
        "precision": "intervention_precision",
        "recall": "intervention_recall",
    }
    raw_df = raw_df.rename(columns=raw_rename)

    # Add missing columns with default values for compatibility
    if "over_corrections" not in raw_df.columns:
        raw_df["over_corrections"] = 0
    if "under_corrections" not in raw_df.columns:
        raw_df["under_corrections"] = 0
    if "num_helper_actions" not in raw_df.columns:
        raw_df["num_helper_actions"] = raw_df["num_interventions"]
    if "goal_inference_accuracy" not in raw_df.columns:
        raw_df["goal_inference_accuracy"] = 0.5

    # Create output directory
    output_dir = Path("results/figures/phase9")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Create plot generator
    plotter = PlotGenerator(agg_df, output_dir, raw_df=raw_df)

    # Generate all plots
    generated = []

    print("\n** Generating Figures **")

    # 1. Detection AUROC (main result)
    print("  [1/15] Detection AUROC bar chart...")
    generated.append(plotter.plot_detection_auroc())

    # 2. Detection AUROC detailed
    print("  [2/15] Detection AUROC detailed with violin plot...")
    generated.append(plotter.plot_detection_auroc_detailed())

    # 3. Detection AUROC by condition
    print("  [3/15] Detection AUROC by condition...")
    generated.append(plotter.plot_detection_auroc_by_condition())

    # 4. Detection latency histogram
    print("  [4/15] Detection latency histogram...")
    generated.append(plotter.plot_detection_latency_histogram())

    # 5. Detection latency CDF
    print("  [5/15] Detection latency CDF...")
    generated.append(plotter.plot_detection_latency_cdf())

    # 6. Detection latency boxplot
    print("  [6/15] Detection latency boxplot...")
    generated.append(plotter.plot_detection_latency_boxplot())

    # 7. Task performance
    print("  [7/15] Task performance...")
    generated.append(plotter.plot_task_performance())

    # 8. Task performance detailed
    print("  [8/15] Task performance detailed...")
    generated.append(plotter.plot_task_performance_detailed())

    # 9. Intervention quality
    print("  [9/15] Intervention quality...")
    generated.append(plotter.plot_intervention_quality())

    # 10. Precision-recall scatter
    print("  [10/15] Precision-recall scatter...")
    generated.append(plotter.plot_intervention_precision_recall_scatter())

    # 11. Intervention timing distribution
    print("  [11/15] Intervention timing distribution...")
    generated.append(plotter.plot_intervention_timing_distribution())

    # 12. Model comparison heatmap
    print("  [12/15] Model comparison heatmap...")
    generated.append(plotter.plot_model_comparison_heatmap())

    # 13. Condition comparison heatmap
    print("  [13/15] Condition comparison heatmap...")
    generated.append(plotter.plot_condition_comparison_heatmap())

    # 14. Statistical significance heatmap
    print("  [14/15] Statistical significance heatmap...")
    generated.append(plotter.plot_statistical_significance_heatmap())

    # 15. Summary figure
    print("  [15/15] Comprehensive summary figure...")
    generated.append(plotter.plot_summary_figure())

    # Filter None
    generated = [p for p in generated if p is not None]

    print(f"\n** Results **")
    print(f"Generated {len(generated)} figures in: {output_dir}")

    for p in generated:
        print(f"  - {p.name}")

    # Also export as PDF for publication
    print("\n** Export Summary **")
    print(f"  PNG figures @300dpi: {output_dir}/*.png")
    print("  For PDF export, use matplotlib's savefig with format='pdf'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
