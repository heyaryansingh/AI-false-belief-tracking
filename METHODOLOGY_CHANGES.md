# Methodology Changes Documentation

**Generated**: 2026-01-20 01:17:04
**Phase**: 10 - Statistical Strengthening
**Total Fixes**: 61

---

## Executive Summary

This document summarizes all methodology changes made during Phase 10 to strengthen
the statistical validity and scientific rigor of the experimental pipeline.

### Key Improvements

1. **Bootstrap Confidence Intervals**: Replaced mean ± SD with mean [95% CI] for AUROC
2. **Effect Size Calculations**: Added Cohen's d for pairwise model comparisons
3. **Temporal Metrics**: Added time-to-detection and false alarm rate tracking
4. **Three Conditions**: Added partial_false_belief (drift_probability=0.5)
5. **Visualization**: Added ROC curves, violin plots, and diagnostic figures
6. **Per-Episode Independence**: Ensured efficiency is computed independently per model/episode

---

## AUROC Stabilization

### Seed logging for full reproducibility

**Location**: `scripts\run_phase9_experiments.py` (line 5)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Enhanced metrics with temporal tracking and bootstrap CIs

**Location**: `scripts\run_phase9_experiments.py` (line 6)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Seed logging for full reproducibility

**Location**: `scripts\run_phase9_experiments.py` (line 86)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Use condition-specific drift probability

**Location**: `scripts\run_phase9_experiments.py` (line 101)

**Phase**: Unknown

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Log seed for reproducibility

**Location**: `scripts\run_phase9_experiments.py` (line 104)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Include seed for reproducibility

**Location**: `scripts\run_phase9_experiments.py` (line 136)

**Phase**: Unknown

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Display statistics with CIs instead of just mean ± SD

**Location**: `scripts\run_phase9_experiments.py` (line 178)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Save seed manifest for reproducibility

**Location**: `scripts\run_phase9_experiments.py` (line 257)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Bootstrap CI replaces SD for robust confidence bounds

**Location**: `src\bsa\analysis\aggregate.py` (line 3)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Bootstrap CI replaces SD for robust confidence bounds

**Location**: `src\bsa\analysis\aggregate.py` (line 322)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Add bootstrap CI for AUROC

**Location**: `src\bsa\analysis\aggregate.py` (line 339)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Add bootstrap CI for latency

**Location**: `src\bsa\analysis\aggregate.py` (line 355)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Bootstrap CI replaces SD for robust confidence bounds

**Location**: `src\bsa\analysis\aggregate.py` (line 573)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Bootstrap CI added to stabilize AUROC variance

**Location**: `src\bsa\analysis\statistics.py` (line 9)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Bootstrap CI added to stabilize AUROC variance - uses resampling

**Location**: `src\bsa\analysis\statistics.py` (line 26)

**Phase**: Unknown

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Generic bootstrap CI function for any metric aggregation.

**Location**: `src\bsa\analysis\statistics.py` (line 125)

**Phase**: Unknown

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### T-distribution CI for small samples (n < 30), normal for larger.

**Location**: `src\bsa\analysis\statistics.py` (line 180)

**Phase**: Unknown

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Comprehensive statistics table with CIs and sample sizes

**Location**: `src\bsa\analysis\tables.py` (line 3)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Comprehensive statistics table with CIs and sample sizes

**Location**: `src\bsa\analysis\tables.py` (line 549)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### ROC curves with CI shading for model comparison

**Location**: `src\bsa\analysis\visualization.py` (line 3)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### AUROC distribution violin plots for variance visualization

**Location**: `src\bsa\analysis\visualization.py` (line 4)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### ROC curves with CI shading for model comparison

**Location**: `src\bsa\analysis\visualization.py` (line 64)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### AUROC distribution violin plots for variance visualization

**Location**: `src\bsa\analysis\visualization.py` (line 148)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### AUROC computed per-episode, aggregated later with bootstrap CI

**Location**: `src\bsa\experiments\evaluator.py` (line 3)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### AUROC computed per-episode, aggregated later with bootstrap CI

**Location**: `src\bsa\experiments\evaluator.py` (line 127)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Per-episode AUROC for later aggregation with bootstrap CI

**Location**: `src\bsa\experiments\evaluator.py` (line 189)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Efficiency recalculated per model/episode independently

**Location**: `src\bsa\experiments\evaluator.py` (line 272)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Temporal precision-recall tracks detection quality over episode

**Location**: `src\bsa\experiments\evaluator.py` (line 369)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

### Compute precision/recall over time (cumulative at each timestep)

**Location**: `src\bsa\experiments\evaluator.py` (line 410)

**Phase**: 10

**Rationale**: High AUROC variance (σ = 0.409) suggests unstable sampling or metric misuse. Bootstrap CI provides robust confidence bounds that are more reliable than simple standard deviation for classification metrics.

---

## Efficiency Fix

### Added detailed wasted action breakdown

**Location**: `scripts\run_phase9_experiments.py` (line 149)

**Phase**: 10

**Rationale**: Identical efficiency across models (0.815) suggested shared calculation or caching. Ensuring per-model/episode independence provides valid comparative metrics.

---

### Ensure wasted_actions is computed independently for each episode

**Location**: `src\bsa\experiments\evaluator.py` (line 287)

**Phase**: 10

**Rationale**: Identical efficiency across models (0.815) suggested shared calculation or caching. Ensuring per-model/episode independence provides valid comparative metrics.

---

### Computed per-episode, not globally

**Location**: `src\bsa\experiments\evaluator.py` (line 333)

**Phase**: 10

**Rationale**: Identical efficiency across models (0.815) suggested shared calculation or caching. Ensuring per-model/episode independence provides valid comparative metrics.

---

### Added detailed wasted action breakdown

**Location**: `src\bsa\experiments\evaluator.py` (line 353)

**Phase**: 10

**Rationale**: Identical efficiency across models (0.815) suggested shared calculation or caching. Ensuring per-model/episode independence provides valid comparative metrics.

---

## Temporal Metrics

### Extended result dictionary with temporal metrics

**Location**: `scripts\run_phase9_experiments.py` (line 130)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Added temporal metrics

**Location**: `scripts\run_phase9_experiments.py` (line 141)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Temporal metrics summary

**Location**: `scripts\run_phase9_experiments.py` (line 240)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Add temporal metrics aggregation

**Location**: `src\bsa\analysis\aggregate.py` (line 371)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Temporal metrics plots for detection timing analysis

**Location**: `src\bsa\analysis\visualization.py` (line 5)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Temporal metrics plots for detection timing analysis

**Location**: `src\bsa\analysis\visualization.py` (line 230)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Added temporal metrics

**Location**: `src\bsa\experiments\evaluator.py` (line 148)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Track all detection timesteps

**Location**: `src\bsa\experiments\evaluator.py` (line 169)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Compute time-to-detection (TTD) - mean time from onset to all detections

**Location**: `src\bsa\experiments\evaluator.py` (line 181)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Added temporal metrics for realistic detection timing

**Location**: `src\bsa\experiments\evaluator.py` (line 221)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

### Added F1 score and temporal metrics

**Location**: `src\bsa\experiments\evaluator.py` (line 440)

**Phase**: 10

**Rationale**: AUROC treats detection as static classification, ignoring temporal dynamics. Adding detection latency and time-to-detection captures realistic detection timing.

---

## Statistical Reporting

### Pairwise comparisons with effect sizes and significance tests

**Location**: `src\bsa\analysis\aggregate.py` (line 465)

**Phase**: 10

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

### Effect size (Cohen's d) for pairwise model comparisons.

**Location**: `src\bsa\analysis\statistics.py` (line 234)

**Phase**: Unknown

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

### Paired t-test for statistical significance testing between models.

**Location**: `src\bsa\analysis\statistics.py` (line 308)

**Phase**: Unknown

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

### Comprehensive statistics for publication-ready reporting.

**Location**: `src\bsa\analysis\statistics.py` (line 531)

**Phase**: Unknown

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

### Pairwise comparisons with effect sizes and significance tests

**Location**: `src\bsa\analysis\tables.py` (line 4)

**Phase**: 10

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

### Pairwise comparisons with effect sizes and significance tests

**Location**: `src\bsa\analysis\tables.py` (line 611)

**Phase**: 10

**Rationale**: Raw means without inferential statistics limit scientific validity. Effect sizes and significance tests enable proper hypothesis testing.

---

## Experimental Design

### Added partial_false_belief condition for intermediate belief state

**Location**: `scripts\run_phase9_experiments.py` (line 4)

**Phase**: 10

**Rationale**: Two-condition design (control, false_belief) lacks intermediate states. Adding partial_false_belief enables more nuanced analysis of model behavior.

---

### Added partial_false_belief condition for intermediate belief state

**Location**: `scripts\run_phase9_experiments.py` (line 77)

**Phase**: 10

**Rationale**: Two-condition design (control, false_belief) lacks intermediate states. Adding partial_false_belief enables more nuanced analysis of model behavior.

---

## Other

### Updated analysis pipeline with new statistical functions

**Location**: `scripts\analyze_results.py` (line 4)

**Phase**: 10

---

### Auto-generate change log from code comments

**Location**: `scripts\generate_change_log.py` (line 4)

**Phase**: 10

---

### " comments and generates

**Location**: `scripts\generate_change_log.py` (line 6)

**Phase**: Unknown

---

### " comments in Python files.

**Location**: `scripts\generate_change_log.py` (line 22)

**Phase**: Unknown

---

### " in line:

**Location**: `scripts\generate_change_log.py` (line 45)

**Phase**: Unknown

---

### ' comments in {args.source_dir}...")

**Location**: `scripts\generate_change_log.py` (line 258)

**Phase**: Unknown

---

### Non-parametric alternative to paired t-test for non-normal data.

**Location**: `src\bsa\analysis\statistics.py` (line 349)

**Phase**: Unknown

---

### Validate scores are probabilities [0, 1]

**Location**: `src\bsa\experiments\evaluator.py` (line 155)

**Phase**: 10

---

### Compute false alarm rate per episode

**Location**: `src\bsa\experiments\evaluator.py` (line 213)

**Phase**: 10

---

## Files Modified

- `scripts\analyze_results.py` (1 changes)
- `scripts\generate_change_log.py` (5 changes)
- `scripts\run_phase9_experiments.py` (14 changes)
- `src\bsa\analysis\aggregate.py` (7 changes)
- `src\bsa\analysis\statistics.py` (8 changes)
- `src\bsa\analysis\tables.py` (4 changes)
- `src\bsa\analysis\visualization.py` (6 changes)
- `src\bsa\experiments\evaluator.py` (16 changes)

## Appendix: All Fix Comments

| File | Line | Description |
|------|------|-------------|
| `scripts\analyze_results.py` | 4 | Updated analysis pipeline with new statistical functions |
| `scripts\generate_change_log.py` | 4 | Auto-generate change log from code comments |
| `scripts\generate_change_log.py` | 6 | " comments and generates |
| `scripts\generate_change_log.py` | 22 | " comments in Python files. |
| `scripts\generate_change_log.py` | 45 | " in line: |
| `scripts\generate_change_log.py` | 258 | ' comments in {args.source_dir}...") |
| `scripts\run_phase9_experiments.py` | 4 | Added partial_false_belief condition for intermediate belief... |
| `scripts\run_phase9_experiments.py` | 5 | Seed logging for full reproducibility |
| `scripts\run_phase9_experiments.py` | 6 | Enhanced metrics with temporal tracking and bootstrap CIs |
| `scripts\run_phase9_experiments.py` | 77 | Added partial_false_belief condition for intermediate belief... |
| `scripts\run_phase9_experiments.py` | 86 | Seed logging for full reproducibility |
| `scripts\run_phase9_experiments.py` | 101 | Use condition-specific drift probability |
| `scripts\run_phase9_experiments.py` | 104 | Log seed for reproducibility |
| `scripts\run_phase9_experiments.py` | 130 | Extended result dictionary with temporal metrics |
| `scripts\run_phase9_experiments.py` | 136 | Include seed for reproducibility |
| `scripts\run_phase9_experiments.py` | 141 | Added temporal metrics |
| `scripts\run_phase9_experiments.py` | 149 | Added detailed wasted action breakdown |
| `scripts\run_phase9_experiments.py` | 178 | Display statistics with CIs instead of just mean ± SD |
| `scripts\run_phase9_experiments.py` | 240 | Temporal metrics summary |
| `scripts\run_phase9_experiments.py` | 257 | Save seed manifest for reproducibility |
| `src\bsa\analysis\aggregate.py` | 3 | Bootstrap CI replaces SD for robust confidence bounds |
| `src\bsa\analysis\aggregate.py` | 322 | Bootstrap CI replaces SD for robust confidence bounds |
| `src\bsa\analysis\aggregate.py` | 339 | Add bootstrap CI for AUROC |
| `src\bsa\analysis\aggregate.py` | 355 | Add bootstrap CI for latency |
| `src\bsa\analysis\aggregate.py` | 371 | Add temporal metrics aggregation |
| `src\bsa\analysis\aggregate.py` | 465 | Pairwise comparisons with effect sizes and significance test... |
| `src\bsa\analysis\aggregate.py` | 573 | Bootstrap CI replaces SD for robust confidence bounds |
| `src\bsa\analysis\statistics.py` | 9 | Bootstrap CI added to stabilize AUROC variance |
| `src\bsa\analysis\statistics.py` | 26 | Bootstrap CI added to stabilize AUROC variance - uses resamp... |
| `src\bsa\analysis\statistics.py` | 125 | Generic bootstrap CI function for any metric aggregation. |
| `src\bsa\analysis\statistics.py` | 180 | T-distribution CI for small samples (n < 30), normal for lar... |
| `src\bsa\analysis\statistics.py` | 234 | Effect size (Cohen's d) for pairwise model comparisons. |
| `src\bsa\analysis\statistics.py` | 308 | Paired t-test for statistical significance testing between m... |
| `src\bsa\analysis\statistics.py` | 349 | Non-parametric alternative to paired t-test for non-normal d... |
| `src\bsa\analysis\statistics.py` | 531 | Comprehensive statistics for publication-ready reporting. |
| `src\bsa\analysis\tables.py` | 3 | Comprehensive statistics table with CIs and sample sizes |
| `src\bsa\analysis\tables.py` | 4 | Pairwise comparisons with effect sizes and significance test... |
| `src\bsa\analysis\tables.py` | 549 | Comprehensive statistics table with CIs and sample sizes |
| `src\bsa\analysis\tables.py` | 611 | Pairwise comparisons with effect sizes and significance test... |
| `src\bsa\analysis\visualization.py` | 3 | ROC curves with CI shading for model comparison |
| `src\bsa\analysis\visualization.py` | 4 | AUROC distribution violin plots for variance visualization |
| `src\bsa\analysis\visualization.py` | 5 | Temporal metrics plots for detection timing analysis |
| `src\bsa\analysis\visualization.py` | 64 | ROC curves with CI shading for model comparison |
| `src\bsa\analysis\visualization.py` | 148 | AUROC distribution violin plots for variance visualization |
| `src\bsa\analysis\visualization.py` | 230 | Temporal metrics plots for detection timing analysis |
| `src\bsa\experiments\evaluator.py` | 3 | AUROC computed per-episode, aggregated later with bootstrap ... |
| `src\bsa\experiments\evaluator.py` | 127 | AUROC computed per-episode, aggregated later with bootstrap ... |
| `src\bsa\experiments\evaluator.py` | 148 | Added temporal metrics |
| `src\bsa\experiments\evaluator.py` | 155 | Validate scores are probabilities [0, 1] |
| `src\bsa\experiments\evaluator.py` | 169 | Track all detection timesteps |
| `src\bsa\experiments\evaluator.py` | 181 | Compute time-to-detection (TTD) - mean time from onset to al... |
| `src\bsa\experiments\evaluator.py` | 189 | Per-episode AUROC for later aggregation with bootstrap CI |
| `src\bsa\experiments\evaluator.py` | 213 | Compute false alarm rate per episode |
| `src\bsa\experiments\evaluator.py` | 221 | Added temporal metrics for realistic detection timing |
| `src\bsa\experiments\evaluator.py` | 272 | Efficiency recalculated per model/episode independently |
| `src\bsa\experiments\evaluator.py` | 287 | Ensure wasted_actions is computed independently for each epi... |
| `src\bsa\experiments\evaluator.py` | 333 | Computed per-episode, not globally |
| `src\bsa\experiments\evaluator.py` | 353 | Added detailed wasted action breakdown |
| `src\bsa\experiments\evaluator.py` | 369 | Temporal precision-recall tracks detection quality over epis... |
| `src\bsa\experiments\evaluator.py` | 410 | Compute precision/recall over time (cumulative at each times... |
| `src\bsa\experiments\evaluator.py` | 440 | Added F1 score and temporal metrics |
