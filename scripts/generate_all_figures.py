#!/usr/bin/env python3
"""Generate all comprehensive figures with detailed visualizations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
from bsa.analysis.aggregate import AnalysisAggregator
from bsa.viz.plots import PlotGenerator

def main():
    """Generate all figures."""
    print("=" * 70)
    print("Generating All Comprehensive Figures")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading data...")
    aggregator = AnalysisAggregator()
    input_dir = Path("results/metrics/large_scale_research")
    raw_df = aggregator.load_results(input_dir=input_dir)
    aggregated_df = aggregator.aggregate_metrics(raw_df)
    
    print(f"  Loaded {len(raw_df)} runs")
    print(f"  Aggregated into {len(aggregated_df)} groups")
    
    # Create plotter
    print("\n[2/3] Creating plot generator...")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotter = PlotGenerator(aggregated_df, figures_dir, raw_df=raw_df)
    
    # Generate all figures
    print("\n[3/3] Generating all figures...")
    generated = []
    
    # Detection plots
    print("  Detection AUROC plots...")
    generated.append(plotter.plot_detection_auroc())
    generated.append(plotter.plot_detection_auroc_detailed())
    generated.append(plotter.plot_detection_auroc_by_condition())
    
    # Latency plots
    print("  Detection latency plots...")
    generated.append(plotter.plot_detection_latency_histogram())
    generated.append(plotter.plot_detection_latency_cdf())
    generated.append(plotter.plot_detection_latency_boxplot())
    
    # Task performance
    print("  Task performance plots...")
    generated.append(plotter.plot_task_performance())
    generated.append(plotter.plot_task_performance_detailed())
    
    # Intervention quality
    print("  Intervention quality plots...")
    generated.append(plotter.plot_intervention_quality())
    generated.append(plotter.plot_intervention_precision_recall_scatter())
    generated.append(plotter.plot_intervention_timing_distribution())
    
    # Belief tracking
    print("  Belief tracking plots...")
    generated.append(plotter.plot_belief_timeline())
    generated.append(plotter.plot_goal_inference_by_condition())
    
    # Heatmaps
    print("  Comparison heatmaps...")
    generated.append(plotter.plot_model_comparison_heatmap())
    generated.append(plotter.plot_condition_comparison_heatmap())
    generated.append(plotter.plot_statistical_significance_heatmap("false_belief_detection_auroc"))
    
    # Ablation
    print("  Ablation plots...")
    generated.append(plotter.plot_tau_effect())
    
    # Summary
    print("  Summary figure...")
    generated.append(plotter.plot_summary_figure())
    
    # Filter None values
    generated = [g for g in generated if g is not None]
    
    print(f"\n[OK] Generated {len(generated)} figures")
    print(f"  Saved to: {figures_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
