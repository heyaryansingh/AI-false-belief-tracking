# Figures and Tables Organization

This document organizes all visualizations and tables for the research paper.

## Figures

### Figure 1: Detection AUROC Detailed
- **File**: `results/figures/detection_auroc_detailed.png`
- **Type**: Detailed AUROC comparison
- **Description**: Shows individual runs, distributions, and statistical comparisons
- **Section**: Results - False-Belief Detection
- **Caption**: "Detailed AUROC comparison showing individual runs and distributions across models in the false-belief condition."

### Figure 2: Task Performance Detailed
- **File**: `results/figures/task_performance_detailed.png`
- **Type**: Task performance comparison
- **Description**: Violin plots and bar charts showing task efficiency, wasted actions, helper actions, and intervention counts
- **Section**: Results - Task Performance
- **Caption**: "Task performance comparison across models showing efficiency, wasted actions, and intervention patterns."

### Figure 3: Intervention Quality
- **File**: `results/figures/intervention_quality_detailed.png`
- **Type**: Intervention quality metrics
- **Description**: Precision/recall comparison and over/under-correction analysis
- **Section**: Results - Intervention Quality
- **Caption**: "Intervention quality metrics showing precision/recall and over/under-correction rates across models."

### Figure 4: Belief Timeline Sample
- **File**: `results/figures/belief_timeline_sample.png`
- **Type**: Belief evolution timeline
- **Description**: Sample belief state evolution over time for selected episodes
- **Section**: Results - Belief Tracking
- **Caption**: "Sample belief evolution timeline showing how belief states change over time during an episode."

### Additional Figures Available

- `detection_auroc.png`: Basic AUROC comparison
- `detection_auroc_by_condition.png`: AUROC by condition
- `detection_latency_histogram.png`: Detection latency distribution
- `detection_latency_cdf.png`: Cumulative distribution of detection latency
- `detection_latency_boxplot.png`: Box plot of detection latency by model
- `goal_inference_by_condition.png`: Goal inference accuracy by condition
- `intervention_pr_scatter.png`: Precision-recall scatter plot
- `intervention_timing_dist.png`: Intervention timing distribution
- `model_comparison_heatmap.png`: Model comparison heatmap
- `condition_comparison_heatmap.png`: Condition comparison heatmap
- `significance_heatmap_false_belief_detection_auroc.png`: Statistical significance heatmap
- `tau_effect.png`: Tau (intervention timing) effect analysis
- `summary_figure.png`: Comprehensive summary figure

## Tables

### Table 1: Summary Statistics
- **File**: `results/tables/summary.md` (Markdown), `results/tables/summary.tex` (LaTeX)
- **Type**: Summary table
- **Description**: Overall comparison of models across key metrics (AUROC, latency, completion, wasted actions, efficiency)
- **Section**: Results - Summary Statistics
- **Caption**: "Summary statistics comparing models across key metrics. Values shown as mean ± standard deviation."

### Table 2: False-Belief Detection Metrics
- **File**: `results/tables/detection.md` (Markdown), `results/tables/detection.tex` (LaTeX)
- **Type**: Detection metrics table
- **Description**: Detailed false-belief detection metrics (AUROC, latency, FPR)
- **Section**: Results - False-Belief Detection
- **Caption**: "False-belief detection metrics across models. AUROC values shown as mean ± standard deviation."

### Table 3: Task Performance Metrics
- **File**: `results/tables/task_performance.md` (Markdown), `results/tables/task_performance.tex` (LaTeX)
- **Type**: Task performance table
- **Description**: Task performance metrics (completion rate, steps, wasted actions, efficiency)
- **Section**: Results - Task Performance
- **Caption**: "Task performance metrics across models. Values shown as mean ± standard deviation."

### Table 4: Intervention Quality Metrics
- **File**: `results/tables/intervention.md` (Markdown), `results/tables/intervention.tex` (LaTeX)
- **Type**: Intervention quality table
- **Description**: Intervention quality metrics (precision, recall, over/under-corrections)
- **Section**: Results - Intervention Quality
- **Caption**: "Intervention quality metrics across models. Precision and recall shown as mean ± standard deviation."

## Figure Selection for Paper

**Primary Figures** (included in main paper):
1. Detection AUROC Detailed (Figure 1)
2. Task Performance Detailed (Figure 2)
3. Intervention Quality (Figure 3)
4. Belief Timeline Sample (Figure 4)

**Supplementary Figures** (for appendix or supplementary material):
- Detection latency analysis (histogram, CDF, boxplot)
- Goal inference by condition
- Intervention timing distribution
- Model/condition comparison heatmaps
- Statistical significance heatmaps
- Tau effect analysis
- Summary figure

## Table Selection for Paper

**Primary Tables** (included in main paper):
1. Summary Statistics (Table 1)
2. False-Belief Detection Metrics (Table 2)
3. Task Performance Metrics (Table 3)
4. Intervention Quality Metrics (Table 4)

All tables are available in both Markdown (for GitHub/documentation) and LaTeX (for paper submission) formats.

## Organization by Section

### Introduction
- No figures/tables

### Related Work
- No figures/tables

### Methodology
- System architecture diagram (if created)
- Particle filter visualization (if created)

### Results
- **Summary Statistics**: Table 1
- **False-Belief Detection**: Figure 1, Table 2
- **Task Performance**: Figure 2, Table 3
- **Intervention Quality**: Figure 3, Table 4
- **Belief Tracking**: Figure 4

### Discussion
- Summary figure (if needed)

### Conclusion
- No figures/tables

## File Paths

All figures are located in: `results/figures/`
All tables are located in: `results/tables/`

For paper submission, figures should be:
- High resolution (300+ DPI)
- Publication-ready format (PNG or PDF)
- Properly captioned and referenced

Tables should be:
- Properly formatted (LaTeX for submission)
- Include statistical annotations
- Clear and readable
