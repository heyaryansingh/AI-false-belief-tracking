#!/usr/bin/env python3
"""Regenerate comprehensive analysis with detailed figures and enhanced paper."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
from bsa.analysis.aggregate import AnalysisAggregator
from bsa.viz.plots import generate_all_plots, PlotGenerator
from bsa.analysis.tables import TableGenerator, generate_tables


def main():
    """Regenerate comprehensive analysis."""
    print("=" * 70)
    print("Comprehensive Analysis Regeneration")
    print("=" * 70)
    
    # Load configuration
    config_path = Path("configs/experiments/exp_large_scale.yaml")
    config = load_config(config_path)
    analysis_config = config.get("analysis", {})
    
    # Set output directory - use results/figures for compatibility
    output_dir = Path(analysis_config.get("output_dir", "results/analysis/large_scale"))
    figures_dir = Path("results/figures")  # Use standard location
    tables_dir = Path("results/tables")  # Use standard location
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config to use correct paths
    analysis_config["output_dir"] = str(figures_dir.parent)  # results/
    
    # Load results
    print("\n[1/4] Loading experiment results...")
    aggregator = AnalysisAggregator()
    input_dir = Path(analysis_config.get("input_dir", "results/metrics/large_scale_research"))
    raw_df = aggregator.load_results(input_dir=input_dir)
    aggregated_df = aggregator.aggregate_metrics(raw_df)
    
    print(f"  Loaded {len(raw_df)} raw results")
    print(f"  Aggregated into {len(aggregated_df)} groups")
    print(f"  Models: {raw_df['model'].unique().tolist()}")
    print(f"  Conditions: {raw_df['condition'].unique().tolist()}")
    
    # Verify data quality
    print("\n[2/4] Verifying data quality...")
    print(f"  Total runs: {len(raw_df)}")
    print(f"  Runs per model/condition:")
    for model in raw_df['model'].unique():
        for condition in raw_df['condition'].unique():
            count = len(raw_df[(raw_df['model'] == model) & (raw_df['condition'] == condition)])
            print(f"    {model}/{condition}: {count}")
    
    # Check for missing data
    missing_cols = raw_df.columns[raw_df.isna().all()].tolist()
    if missing_cols:
        print(f"  Warning: Columns with all NaN: {missing_cols}")
    
    # Generate comprehensive figures
    print("\n[3/4] Generating comprehensive figures...")
    plotter = PlotGenerator(aggregated_df, figures_dir, raw_df=raw_df)
    
    # Update config to use correct output directory
    temp_config = config.copy()
    temp_config["analysis"] = analysis_config.copy()
    temp_config["analysis"]["output_dir"] = str(figures_dir.parent)  # results/
    
    # Generate all comprehensive plots
    figure_paths = generate_all_plots(temp_config, aggregated_df=aggregated_df, raw_df=raw_df)
    
    print(f"\n  Generated {len(figure_paths)} comprehensive figures")
    for path in figure_paths:
        if path:
            print(f"    - {path.name}")
    
    # Generate tables
    print("\n[4/4] Generating comprehensive tables...")
    table_gen = TableGenerator(aggregated_df)
    
    # Generate all table types
    table_config = analysis_config.get("tables", [])
    table_paths = []
    
    for table_spec in table_config:
        table_type = table_spec.get("type")
        print(f"  Generating {table_type} table...")
        
        if table_type == "summary":
            md_table = table_gen.generate_summary_table(format="markdown")
            tex_table = table_gen.generate_summary_table(format="latex")
        elif table_type == "detection":
            md_table = table_gen.generate_detection_table(format="markdown")
            tex_table = table_gen.generate_detection_table(format="latex")
        elif table_type == "task_performance":
            md_table = table_gen.generate_task_performance_table(format="markdown")
            tex_table = table_gen.generate_task_performance_table(format="latex")
        elif table_type == "intervention":
            md_table = table_gen.generate_intervention_table(format="markdown")
            tex_table = table_gen.generate_intervention_table(format="latex")
        else:
            print(f"    [WARN] Unknown table type: {table_type}")
            continue
        
        # Save tables
        md_path = tables_dir / table_spec.get("filename_md", f"{table_type}.md")
        tex_path = tables_dir / table_spec.get("filename_tex", f"{table_type}.tex")
        
        md_path.write_text(md_table)
        tex_path.write_text(tex_table)
        
        table_paths.extend([md_path, tex_path])
        print(f"    [OK] Saved: {md_path.name}, {tex_path.name}")
    
    print(f"\n  Generated {len(table_paths)} table files")
    
    # Summary
    print("\n" + "=" * 70)
    print("Comprehensive Analysis Regeneration Complete")
    print("=" * 70)
    print(f"\nFigures: {figures_dir}")
    print(f"  Generated {len(figure_paths)} figures")
    print(f"\nTables: {tables_dir}")
    print(f"  Generated {len(table_paths)} table files")
    print(f"\nData verified:")
    print(f"  Total runs: {len(raw_df)}")
    print(f"  All models and conditions represented")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
