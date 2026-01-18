# Roadmap

## Phase 1: Core Interfaces + GridHouse Fallback Simulator

**Goal**: Complete GridHouse simulator with episode generation and human agent policies

**Status**: Complete

**Tasks**:
- Complete episode generator with belief tracking and intervention logic
- Implement human agent scripted policies
- Add episode serialization (Parquet/JSONL)
- Verify GridHouse end-to-end with test episodes

**Research**: No

**Dependencies**: None

---

## Phase 2: Helper Models (Baselines + Belief-PF)

**Goal**: Implement reactive, goal-only, and belief-sensitive helper agents

**Status**: Complete

**Tasks**:
- Base helper interface
- Reactive helper (baseline)
- Goal-only helper (baseline)
- Belief particle filter helper (main contribution)
- Intervention policy
- Goal inference module
- Belief inference module
- Particle filter implementation
- Likelihood models (rule-based, learned)

**Research**: No

**Dependencies**: Phase 1

---

## Phase 3: VirtualHome Backend

**Goal**: Implement VirtualHome adapter with task programs and observability

**Status**: Complete

**Tasks**:
- VirtualHome adapter implementation
- Task programs library
- Observability module
- Episode generator for VirtualHome
- Recorder for episodes
- Installation script and verification
- VirtualHome-specific tests
- End-to-end verification

**Research**: Yes (VirtualHome API research needed)

**Dependencies**: Phase 1

---

## Phase 4: Experiment Harness + Reproducibility

**Goal**: Automated experiment runner with result saving and manifests

**Status**: Complete

**Tasks**:
- Experiment runner implementation
- Sweep runner for ablations
- Episode evaluator
- Reproducibility scripts
- Manifest generation (git hash, config hash, versions)
- CLI command completion

**Research**: No

**Dependencies**: Phase 2, Phase 3

---

## Phase 5: Metrics + Analysis + Report

**Goal**: Comprehensive metrics, plots, tables, and technical report generation

**Status**: Pending

**Tasks**:
- Drift detection metrics (AUROC, detection delay, FPR)
- Belief tracking metrics (accuracy, cross-entropy, Brier score)
- Task performance metrics (completion, wasted actions)
- Intervention quality metrics (over/under-correction, precision/recall)
- Analysis aggregation
- Plotting (timelines, histograms, ablation curves)
- Table generation
- Technical report generation

**Research**: No

**Dependencies**: Phase 4

---

## Phase 6: Tests + CI

**Goal**: Comprehensive test suite and CI verification

**Status**: Complete

**Tasks**:
- Episode generator tests ✅
- Particle filter tests ✅
- Inference module tests ✅
- Helper agent tests ✅
- Experiment component tests ✅
- Analysis component tests ✅
- Integration tests ✅
- VirtualHome installation verification ✅
- CI workflow completion ✅

**Research**: No

**Dependencies**: All previous phases

---

## Phase 7: Research Execution & Paper Writing

**Goal**: Execute comprehensive experiments, collect extensive data, generate detailed visualizations, and write research paper

**Status**: In Progress (Plan 2 of 3 complete)

**Tasks**:
- Plan 07-01: Large-scale experiment execution - COMPLETE
  - 9,960 episodes generated
  - 450 experiment runs (3 models x 3 conditions x 50 runs)
- Plan 07-02: Comprehensive analysis & visualization - COMPLETE
  - Enhanced plotting module with 18 visualization types
  - Generated 18 publication-quality figures
  - Generated 8 tables (Markdown + LaTeX)
  - Statistical significance analysis included
- Plan 07-03: Paper writing - Ready

**Research**: Yes - Core research phase

**Dependencies**: Phase 6 (all infrastructure complete)

**Research Question**: Can belief-sensitive assistance (using particle filter/Bayesian inference) outperform reactive and goal-only baselines on false-belief detection, task completion, and wasted action reduction?

**Target**: Labs, jobs, and AI researchers
