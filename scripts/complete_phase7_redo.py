#!/usr/bin/env python3
"""Complete Phase 7 redo: Regenerate everything from scratch with improvements."""

import sys
from pathlib import Path
import shutil
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
    """Complete Phase 7 redo."""
    print("=" * 70)
    print("Phase 7 Complete Redo: Regenerating Everything from Scratch")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Clean old data
    print("[1] Cleaning old data...")
    old_dirs = [
        Path("data/episodes/large_scale"),
        Path("results/metrics/large_scale_research"),
        Path("results/analysis/large_scale"),
        Path("results/figures"),
        Path("results/tables"),
    ]
    
    for dir_path in old_dirs:
        if dir_path.exists():
            print(f"  Removing {dir_path}...")
            shutil.rmtree(dir_path)
    
    # Recreate directories
    for dir_path in old_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("  [OK] Old data cleaned")
    
    # Step 2: Load config
    print("\n[2] Loading configuration...")
    config_path = Path("configs/experiments/exp_large_scale.yaml")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    exp_config = config.get("experiment", {})
    gen_config = config.get("generator", {})
    
    print(f"  Episodes: {gen_config.get('num_episodes', 0)}")
    print(f"  Runs per config: {exp_config.get('num_runs', 0)}")
    print(f"  Models: {exp_config.get('models', [])}")
    print(f"  Conditions: {exp_config.get('conditions', [])}")
    
    # Step 3: Generate episodes
    print("\n[3] Generating episodes from scratch...")
    print("  This will take 30-60 minutes...")
    
    start_time = time.time()
    try:
        episodes = generate_episodes(config, seed=42)
        elapsed = time.time() - start_time
        print(f"  [OK] Generated {len(episodes)} episodes in {elapsed/60:.1f} minutes")
    except Exception as e:
        print(f"  [ERROR] Episode generation failed: {e}")
        return 1
    
    # Step 4: Run experiments
    print("\n[4] Running experiments...")
    print("  This will take 2-4 hours...")
    
    start_time = time.time()
    try:
        results = run_experiments(config, seed=42)
        elapsed = time.time() - start_time
        print(f"  [OK] Completed {results.get('num_results', 0)} runs in {elapsed/60:.1f} minutes")
    except Exception as e:
        print(f"  [ERROR] Experiment execution failed: {e}")
        return 1
    
    # Step 5: Aggregate and analyze
    print("\n[5] Aggregating results...")
    aggregator = AnalysisAggregator()
    results_path = Path("results/metrics/large_scale_research/results.parquet")
    
    if not results_path.exists():
        print(f"  [ERROR] Results file not found: {results_path}")
        return 1
    
    raw_df = aggregator.load_results(input_path=results_path)
    print(f"  Loaded {len(raw_df)} runs")
    
    # Aggregate
    agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
    agg_by_model_cond = aggregator.aggregate_metrics(raw_df, group_by=["model", "condition"])
    
    fb_df = raw_df[raw_df["condition"] == "false_belief"]
    agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])
    
    print(f"  Aggregated by model: {len(agg_by_model)} groups")
    print(f"  Aggregated by model+condition: {len(agg_by_model_cond)} groups")
    
    # Step 6: Generate tables
    print("\n[6] Generating tables...")
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
    
    print("  [OK] Tables generated")
    
    # Step 7: Generate figures
    print("\n[7] Generating figures...")
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
    
    # Step 8: Print summary statistics
    print("\n[8] Summary Statistics:")
    print(f"  Total runs: {len(raw_df)}")
    print(f"  Task completion rate: {raw_df['task_completed'].mean()*100:.2f}%")
    
    # Check AUROC diversity
    auroc_data = raw_df[raw_df["false_belief_detection_auroc"].notna()]["false_belief_detection_auroc"]
    if len(auroc_data) > 0:
        print(f"  AUROC range: {auroc_data.min():.3f} - {auroc_data.max():.3f}")
        print(f"  AUROC mean: {auroc_data.mean():.3f} ± {auroc_data.std():.3f}")
        print(f"  AUROC unique values: {len(auroc_data.unique())}")
    
    print("\n" + "=" * 70)
    print("Phase 7 Redo Complete!")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
