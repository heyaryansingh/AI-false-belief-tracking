# Belief-Sensitive Embodied Assistance: Technical Report

## Abstract

This report presents results from experiments evaluating belief-sensitive embodied assistance systems under object-centered false belief scenarios. We compare reactive, goal-only, and belief-sensitive (particle filter) helper agents across multiple conditions and tasks.

## 1. Introduction

### 1.1 Motivation

Belief-sensitive assistance requires understanding not just what a human is trying to do (goal inference), but also what they believe about the world (belief inference). This is particularly important in scenarios with partial observability, where false beliefs can arise.

### 1.2 Research Questions

1. Can belief-sensitive assistance outperform reactive and goal-only baselines?
2. How accurately can false beliefs be detected?
3. What is the impact on task performance and efficiency?

## 2. Methodology

### 2.1 Tasks

{{TASK_DESCRIPTIONS}}

### 2.2 Helper Agents

**Reactive Baseline**: Reacts to visible objects without inference.

**Goal-Only Baseline**: Infers human goal but assumes beliefs match true state.

**Belief-Sensitive (Particle Filter)**: Tracks both goal and object location beliefs using a particle filter.

### 2.3 Evaluation Metrics

- **False-Belief Detection**: AUROC, detection latency, false positive rate
- **Belief Tracking**: Goal inference accuracy, cross-entropy, Brier score
- **Task Performance**: Completion rate, steps to completion, wasted actions
- **Intervention Quality**: Precision/recall, over/under-correction

### 2.4 Experimental Setup

{{EXPERIMENTAL_SETUP}}

## 3. Results

### 3.1 Summary Statistics

{{SUMMARY_STATS}}

### 3.2 False-Belief Detection

{{DETECTION_RESULTS}}

### 3.3 Task Performance

{{TASK_PERFORMANCE_RESULTS}}

### 3.4 Intervention Quality

{{INTERVENTION_RESULTS}}

### 3.5 Figures

{{FIGURES}}

### 3.6 Tables

{{TABLES}}

## 4. Discussion

{{DISCUSSION}}

### 4.1 Key Findings

{{KEY_FINDINGS}}

### 4.2 Limitations

{{LIMITATIONS}}

## 5. Conclusion

{{CONCLUSION}}

## Appendix

### A. Experimental Details

{{EXPERIMENTAL_DETAILS}}

### B. Hyperparameters

{{HYPERPARAMETERS}}

### C. Reproducibility

{{REPRODUCIBILITY}}
