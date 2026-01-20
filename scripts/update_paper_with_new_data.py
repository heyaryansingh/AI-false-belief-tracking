#!/usr/bin/env python3
"""Update paper with new data from phase7_fixed results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from bsa.analysis.aggregate import AnalysisAggregator
from bsa.analysis.tables import TableGenerator

# Load results
results_path = Path("results/metrics/phase7_fixed/results.parquet")
if not results_path.exists():
    print(f"Error: Results file not found: {results_path}")
    sys.exit(1)

print("Loading results...")
aggregator = AnalysisAggregator()
raw_df = aggregator.load_results(input_path=results_path)
print(f"Loaded {len(raw_df)} runs")

# Aggregate
agg_by_model = aggregator.aggregate_metrics(raw_df, group_by=["model"])
fb_df = raw_df[raw_df["condition"] == "false_belief"]
agg_fb = aggregator.aggregate_metrics(fb_df, group_by=["model"])

# Generate tables
tables_dir = Path("results/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

table_gen_summary = TableGenerator(agg_by_model)
table_gen_fb = TableGenerator(agg_fb)

# Save tables
(tables_dir / "summary.md").write_text(table_gen_summary.generate_summary_table(format="markdown"))
(tables_dir / "summary.tex").write_text(table_gen_summary.generate_summary_table(format="latex"))
(tables_dir / "detection.md").write_text(table_gen_fb.generate_detection_table(format="markdown"))
(tables_dir / "detection.tex").write_text(table_gen_fb.generate_detection_table(format="latex"))
(tables_dir / "task_performance.md").write_text(table_gen_summary.generate_task_performance_table(format="markdown"))
(tables_dir / "task_performance.tex").write_text(table_gen_summary.generate_task_performance_table(format="latex"))
(tables_dir / "intervention.md").write_text(table_gen_fb.generate_intervention_table(format="markdown"))
(tables_dir / "intervention.tex").write_text(table_gen_fb.generate_intervention_table(format="latex"))

print("Tables generated successfully!")

# Print summary
print("\nSummary Statistics:")
print(f"  Total runs: {len(raw_df)}")
print(f"  Task completion: {raw_df['task_completed'].mean()*100:.2f}%")
auroc_data = raw_df[raw_df["false_belief_detection_auroc"].notna()]["false_belief_detection_auroc"]
if len(auroc_data) > 0:
    print(f"  AUROC: {auroc_data.mean():.3f} ± {auroc_data.std():.3f}")
    print(f"  AUROC range: {auroc_data.min():.3f} - {auroc_data.max():.3f}")
    print(f"  AUROC unique values: {len(auroc_data.unique())}")
