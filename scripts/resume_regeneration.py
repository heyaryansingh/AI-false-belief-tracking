#!/usr/bin/env python3
"""Resume regeneration from where it left off."""

import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
from bsa.experiments.run_experiment import generate_episodes, run_experiments
from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
from bsa.viz.plots import PlotGenerator
import pandas as pd

def main():
    """Resume regeneration."""
    print("=" * 70)
    print("Resuming Phase 7 Regeneration")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check current status
    episodes_dir = Path("data/episodes/large_scale")
    existing_episodes = len(list(episodes_dir.glob("*.parquet"))) if episodes_dir.exists() else 0
    
    print(f"[Status] Existing episodes: {existing_episodes} / 10000")
    
    # Load config
    config_path = Path("configs/experiments/exp_large_scale.yaml")
    config = load_config(config_path)
    
    # If episodes incomplete but close enough, proceed anyway
    if existing_episodes < 10000:
        if existing_episodes >= 9000:
            print(f"\n[1] Using existing {existing_episodes} episodes (sufficient for experiments)")
        else:
            print(f"\n[1] Finishing episode generation ({existing_episodes} -> 10000)...")
            gen_config = config.get("generator", {})
            remaining = 10000 - existing_episodes
            gen_config["num_episodes"] = remaining
            
            # Temporarily modify config
            config["generator"] = gen_config
            
            try:
                episodes = generate_episodes(config, seed=42)
                print(f"  [OK] Generated {len(episodes)} additional episodes")
            except Exception as e:
                print(f"  [ERROR] Episode generation failed: {e}")
                print(f"  [INFO] Proceeding with {existing_episodes} episodes anyway")
    
    # Check if experiments need to run
    results_path = Path("results/metrics/large_scale_research/results.parquet")
    if not results_path.exists() or len(pd.read_parquet(results_path)) < 450:
        print("\n[2] Running experiments...")
        try:
            results = run_experiments(config)
            print(f"  [OK] Completed experiments")
        except Exception as e:
            print(f"  [ERROR] Experiment execution failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n[2] Experiments already complete")
    
    # Generate analysis
    print("\n[3] Generating analysis...")
    aggregator = AnalysisAggregator()
    raw_df = aggregator.load_results(input_path=results_path)
    
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    agg_by_model_cond = aggregator.aggregate_metrics(raw_df, group_by=["model", "condition"])
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
    agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    # Tables
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_gen_summary = TableGenerator(agg_by_model)
    table_gen_fb = TableGenerator(agg_fb)
    
    (tables_dir / "summary.md").write_text(table_gen_summary.generate_summary_table(format="markdown"))
    (tables_dir / "summary.tex").write_text(table_gen_summary.generate_summary_table(format="latex"))
    (tables_dir / "detection.md").write_text(table_gen_fb.generate_detection_table(format="markdown"))
    (tables_dir / "detection.tex").write_text(table_gen_fb.generate_detection_table(format="latex"))
    (tables_dir / "task_performance.md").write_text(table_gen_summary.generate_task_performance_table(format="markdown"))
    (tables_dir / "task_performance.tex").write_text(table_gen_summary.generate_task_performance_table(format="latex"))
    (tables_dir / "intervention.md").write_text(table_gen_fb.generate_intervention_table(format="markdown"))
    (tables_dir / "intervention.tex").write_text(table_gen_fb.generate_intervention_table(format="latex"))
    
    # Figures
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_gen = PlotGenerator(agg_by_model_cond, raw_df=raw_df, output_dir=figures_dir)
    
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
    
    print("\n" + "=" * 70)
    print("Regeneration Complete!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
