#!/usr/bin/env python3
"""Comprehensive analysis script for Phase 10 results.

# Fix: Updated analysis pipeline with new statistical functions (Phase 10)

This script:
1. Loads results from parquet
2. Generates statistics tables (with CIs)
3. Generates pairwise comparison tables
4. Generates all diagnostic plots
5. Creates summary report
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator
from bsa.analysis.statistics import (
    compute_bootstrap_ci,
    effect_size,
    format_ci,
    format_p_value,
    independent_ttest,
)
from bsa.analysis.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results with statistical rigor (Phase 10)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("results/metrics/phase10_validation/results.parquet"),
        help="Input parquet file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--generate-tables",
        action="store_true",
        default=True,
        help="Generate statistics tables"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table generation"
    )
    args = parser.parse_args()
    
    # Override flags
    if args.no_plots:
        args.generate_plots = False
    if args.no_tables:
        args.generate_tables = False
    
    print("=" * 70)
    print("Phase 10: Statistical Analysis Pipeline")
    print("=" * 70)
    
    # Check input file
    if not args.input.exists():
        print(f"\nError: Input file not found: {args.input}")
        print("Run experiments first with: python scripts/run_phase9_experiments.py")
        return 1
    
    # Create output directories
    figures_dir = args.output_dir / "figures_v2"
    tables_dir = args.output_dir / "tables_v2"
    reports_dir = args.output_dir / "reports"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"\n[1/5] Loading results from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows")
    print(f"  Models: {df['model'].unique().tolist() if 'model' in df.columns else 'N/A'}")
    print(f"  Conditions: {df['condition'].unique().tolist() if 'condition' in df.columns else 'N/A'}")
    
    # Generate summary statistics
    print("\n[2/5] Computing summary statistics with bootstrap CIs...")
    summary = compute_summary_with_ci(df)
    print_summary(summary)
    
    # Generate tables
    if args.generate_tables:
        print("\n[3/5] Generating statistics tables...")
        table_paths = generate_tables(df, tables_dir)
        print(f"  Generated {len(table_paths)} table files")
    else:
        print("\n[3/5] Skipping table generation")
        table_paths = []
    
    # Generate plots
    if args.generate_plots:
        print("\n[4/5] Generating visualization figures...")
        figure_paths = generate_all_figures(df, figures_dir)
        print(f"  Generated {len(figure_paths)} figures")
    else:
        print("\n[4/5] Skipping plot generation")
        figure_paths = []
    
    # Generate report
    print("\n[5/5] Generating summary report...")
    report_path = generate_report(df, summary, reports_dir, table_paths, figure_paths)
    print(f"  Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Tables:  {tables_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Report:  {report_path}")
    
    return 0


def compute_summary_with_ci(df: pd.DataFrame) -> dict:
    """Compute summary statistics with bootstrap CIs.
    
    Args:
        df: Results DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "models": {},
        "conditions": {},
        "pairwise": [],
    }
    
    models = df["model"].unique() if "model" in df.columns else []
    
    # Per-model statistics
    for model in models:
        model_df = df[df["model"] == model]
        summary["models"][model] = {}
        
        for metric in ["auroc", "efficiency", "precision", "recall", "time_to_detection"]:
            if metric in model_df.columns:
                values = model_df[metric].dropna().values
                if len(values) > 0:
                    ci_result = compute_bootstrap_ci(values, n_bootstrap=1000)
                    summary["models"][model][metric] = {
                        "mean": ci_result["value"],
                        "ci_lower": ci_result["ci_lower"],
                        "ci_upper": ci_result["ci_upper"],
                        "std": ci_result["std"],
                        "n": ci_result["n"],
                    }
    
    # Pairwise comparisons
    model_list = list(models)
    for i, model1 in enumerate(model_list):
        for model2 in model_list[i+1:]:
            for metric in ["auroc", "efficiency"]:
                if metric not in df.columns:
                    continue
                
                values1 = df[df["model"] == model1][metric].dropna().values
                values2 = df[df["model"] == model2][metric].dropna().values
                
                if len(values1) == 0 or len(values2) == 0:
                    continue
                
                d = effect_size(values1, values2)
                ttest = independent_ttest(values1, values2)
                
                summary["pairwise"].append({
                    "model1": model1,
                    "model2": model2,
                    "metric": metric,
                    "effect_size": d,
                    "p_value": ttest["p_value"],
                    "significant": ttest["significant"],
                })
    
    return summary


def print_summary(summary: dict):
    """Print summary statistics to console.
    
    Args:
        summary: Summary dictionary
    """
    print(f"  Total observations: {summary['total_rows']}")
    
    print("\n  Model Statistics (with 95% CI):")
    for model, metrics in summary["models"].items():
        print(f"\n    {model}:")
        for metric, stats in metrics.items():
            ci_str = format_ci(stats["mean"], stats["ci_lower"], stats["ci_upper"])
            print(f"      {metric}: {ci_str} (N={stats['n']})")
    
    print("\n  Pairwise Comparisons:")
    for comp in summary["pairwise"]:
        sig = "*" if comp["significant"] else ""
        d = comp["effect_size"]
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"    {comp['model1']} vs {comp['model2']} ({comp['metric']}): d={d:.3f} ({interp}), p={comp['p_value']:.4f}{sig}")


def generate_tables(df: pd.DataFrame, output_dir: Path) -> list:
    """Generate all statistics tables.
    
    Args:
        df: Results DataFrame
        output_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    # Create aggregated DataFrame for TableGenerator
    aggregator = AnalysisAggregator()
    agg_df = aggregator.aggregate_metrics(df)
    
    table_gen = TableGenerator(agg_df)
    
    paths = []
    
    # Summary table
    summary_md = table_gen.generate_summary_table(format="markdown")
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_md)
    paths.append(summary_path)
    
    # Detection table
    detection_md = table_gen.generate_detection_table(format="markdown")
    detection_path = output_dir / "detection.md"
    detection_path.write_text(detection_md)
    paths.append(detection_path)
    
    # Task performance table
    task_md = table_gen.generate_task_performance_table(format="markdown")
    task_path = output_dir / "task_performance.md"
    task_path.write_text(task_md)
    paths.append(task_path)
    
    # Intervention table
    intervention_md = table_gen.generate_intervention_table(format="markdown")
    intervention_path = output_dir / "intervention.md"
    intervention_path.write_text(intervention_md)
    paths.append(intervention_path)
    
    # New Phase 10 tables
    # Summary statistics with CIs
    stats_md = table_gen.generate_summary_statistics(df, format="markdown")
    stats_path = output_dir / "summary_statistics_ci.md"
    stats_path.write_text(stats_md)
    paths.append(stats_path)
    
    # Pairwise comparisons
    pairwise_md = table_gen.generate_pairwise_comparisons(df, format="markdown")
    pairwise_path = output_dir / "pairwise_comparisons.md"
    pairwise_path.write_text(pairwise_md)
    paths.append(pairwise_path)
    
    # Condition comparison
    condition_md = table_gen.generate_condition_comparison(df, format="markdown")
    condition_path = output_dir / "condition_comparison.md"
    condition_path.write_text(condition_md)
    paths.append(condition_path)
    
    return paths


def generate_report(
    df: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    table_paths: list,
    figure_paths: list,
) -> Path:
    """Generate comprehensive Markdown report.
    
    Args:
        df: Results DataFrame
        summary: Summary statistics
        output_dir: Output directory
        table_paths: Paths to generated tables
        figure_paths: Paths to generated figures
        
    Returns:
        Path to generated report
    """
    lines = [
        "# Phase 10: Statistical Strengthening - Analysis Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"This report summarizes the experimental results after Phase 10 statistical strengthening.",
        "",
        f"- **Total Observations**: {summary['total_rows']}",
        f"- **Models Evaluated**: {', '.join(summary['models'].keys())}",
        f"- **Conditions**: {', '.join(df['condition'].unique()) if 'condition' in df.columns else 'N/A'}",
        "",
        "## Key Results",
        "",
        "### Model Performance (AUROC with 95% CI)",
        "",
        "| Model | AUROC | 95% CI | N |",
        "|-------|-------|--------|---|",
    ]
    
    for model, metrics in summary["models"].items():
        if "auroc" in metrics:
            stats = metrics["auroc"]
            ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
            lines.append(f"| {model} | {stats['mean']:.3f} | {ci} | {stats['n']} |")
    
    lines.extend([
        "",
        "### Effect Sizes (Cohen's d)",
        "",
        "| Comparison | Metric | Effect Size | Interpretation | p-value |",
        "|------------|--------|-------------|----------------|---------|",
    ])
    
    for comp in summary["pairwise"]:
        d = comp["effect_size"]
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        p_str = format_p_value(comp["p_value"])
        lines.append(f"| {comp['model1']} vs {comp['model2']} | {comp['metric']} | {d:.3f} | {interp} | {p_str} |")
    
    lines.extend([
        "",
        "## Methodology Improvements (Phase 10)",
        "",
        "1. **Bootstrap Confidence Intervals**: All metrics now report 95% CIs instead of mean ± SD",
        "2. **Effect Size Calculations**: Cohen's d for pairwise model comparisons",
        "3. **Temporal Metrics**: Time-to-detection and false alarm rate tracking",
        "4. **Three Conditions**: Added partial_false_belief condition",
        "5. **Statistical Tests**: Independent t-tests with significance indicators",
        "",
        "## Generated Outputs",
        "",
        "### Tables",
        "",
    ])
    
    for path in table_paths:
        lines.append(f"- [{path.name}]({path.relative_to(output_dir.parent)})")
    
    lines.extend([
        "",
        "### Figures",
        "",
    ])
    
    for path in figure_paths:
        lines.append(f"- [{path.name}]({path.relative_to(output_dir.parent)})")
    
    lines.extend([
        "",
        "---",
        "",
        "*Report generated by Phase 10 analysis pipeline*",
    ])
    
    report_path = output_dir / "methodology_fixes_report.md"
    report_path.write_text("\n".join(lines))
    
    return report_path


if __name__ == "__main__":
    sys.exit(main())
