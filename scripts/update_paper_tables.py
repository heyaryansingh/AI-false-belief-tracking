#!/usr/bin/env python3
"""Update paper tables with regenerated data."""

from pathlib import Path

# Read regenerated tables
tables_dir = Path("results/tables")

# Read summary table
summary_md = (tables_dir / "summary.md").read_text()
detection_md = (tables_dir / "detection.md").read_text()
task_md = (tables_dir / "task_performance.md").read_text()
intervention_md = (tables_dir / "intervention.md").read_text()

# Extract table rows
def extract_table_rows(md_content):
    """Extract table rows from markdown."""
    lines = md_content.split('\n')
    rows = []
    in_table = False
    for line in lines:
        if '|' in line and not line.strip().startswith('##'):
            if '---' in line:
                in_table = True
                continue
            if in_table:
                rows.append(line.strip())
    return rows[1:] if rows else []  # Skip header

summary_rows = extract_table_rows(summary_md)
detection_rows = extract_table_rows(detection_md)
task_rows = extract_table_rows(task_md)
intervention_rows = extract_table_rows(intervention_md)

print("Summary table rows:")
for row in summary_rows:
    print(f"  {row}")

print("\nDetection table rows:")
for row in detection_rows:
    print(f"  {row}")

print("\nTask performance table rows:")
for row in task_rows:
    print(f"  {row}")

print("\nIntervention table rows:")
for row in intervention_rows:
    print(f"  {row}")

# Update paper
paper_path = Path("paper/research_paper.md")
paper_content = paper_path.read_text()

# Replace Table 1
old_table1 = """| Model | AUROC | Detection Latency | Task Completion | Wasted Actions | Efficiency |
|-------|-------|-------------------|-----------------|----------------|------------|
| belief_pf | 0.500 ± 0.000 | 0.000 ± 0.000 | N/A | 0.000 ± 0.000 | 1.000 ± 0.000 |
| goal_only | 0.500 ± 0.000 | N/A | N/A | 0.020 ± 0.141 | 1.000 ± 0.003 |
| reactive | 0.500 ± 0.000 | N/A | N/A | 0.000 ± 0.000 | 1.000 ± 0.000 |"""

new_table1 = "\n".join(summary_rows)
paper_content = paper_content.replace(old_table1, new_table1)

# Replace Table 3
old_table3 = """| Model | Completion Rate | Steps | Wasted Actions | Efficiency |
|-------|----------------|-------|----------------|------------|
| belief_pf | N/A | N/A | 0.0 ± 0.0 | 1.000 ± 0.000 |
| goal_only | N/A | N/A | 0.0 ± 0.1 | 1.000 ± 0.003 |
| reactive | N/A | N/A | 0.0 ± 0.0 | 1.000 ± 0.000 |"""

new_table3 = "\n".join(task_rows)
paper_content = paper_content.replace(old_table3, new_table3)

paper_path.write_text(paper_content)
print("\n[OK] Paper tables updated!")
