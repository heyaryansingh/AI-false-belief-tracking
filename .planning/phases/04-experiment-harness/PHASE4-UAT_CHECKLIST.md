# Phase 4: Experiment Harness + Reproducibility - User Acceptance Testing Checklist

## Test Scope
Testing all Phase 4 deliverables: Experiment Runner, Episode Evaluator, Sweep Runner, Reproducibility, and CLI completion.

---

## Test 1: Experiment Runner Core (04-01)

### 1.1 Import and Instantiation
- [ ] Can import ExperimentRunner: `from src.bsa.experiments import ExperimentRunner`
- [ ] Can create ExperimentRunner with valid config
- [ ] ExperimentRunner initializes without errors

### 1.2 Episode Generation
- [ ] CLI `generate` command works: `bsa generate --config configs/generator/default.yaml --output data/episodes/test`
- [ ] Episodes are generated and saved
- [ ] Generated episodes are valid (can be loaded)

### 1.3 Experiment Execution
- [ ] Can run experiments with reactive helper
- [ ] Can run experiments with goal_only helper
- [ ] Can run experiments with belief_pf helper
- [ ] Results are saved to `results/metrics/`
- [ ] Results include expected metrics

---

## Test 2: Episode Evaluator (04-02)

### 2.1 Import and Basic Usage
- [ ] Can import EpisodeEvaluator: `from src.bsa.experiments import EpisodeEvaluator`
- [ ] Can create EpisodeEvaluator instance
- [ ] Can evaluate an episode with a helper agent

### 2.2 Metrics Computation
- [ ] False-belief detection metrics computed (AUROC, latency, FPR)
- [ ] Belief tracking metrics computed (accuracy)
- [ ] Task performance metrics computed (completion, wasted actions, efficiency)
- [ ] Intervention metrics computed (over/under-correction, precision/recall)

### 2.3 Integration with ExperimentRunner
- [ ] ExperimentRunner uses EpisodeEvaluator automatically
- [ ] All metrics from EpisodeEvaluator are included in results

---

## Test 3: Sweep Runner & Ablations (04-03)

### 3.1 Import and Configuration
- [ ] Can import SweepRunner: `from src.bsa.experiments import SweepRunner`
- [ ] Sweep config files exist and are valid YAML
- [ ] Can create SweepRunner with sweep config

### 3.2 Single Parameter Sweep
- [ ] Can run single parameter sweep
- [ ] Results are aggregated correctly
- [ ] Results saved to `results/sweeps/`

### 3.3 CLI Integration
- [ ] CLI `sweep` command works: `bsa sweep --config configs/experiments/sweep_particles.yaml`
- [ ] Sweep command produces output
- [ ] Sweep results are saved correctly

---

## Test 4: Reproducibility & Manifests (04-04)

### 4.1 Manifest Generation
- [ ] Can import manifest functions: `from src.bsa.experiments import generate_manifest, save_manifest`
- [ ] Can generate manifest for an experiment
- [ ] Manifest includes git hash, config hash, package versions, system info
- [ ] Manifest is saved correctly

### 4.2 Automatic Manifest Generation
- [ ] ExperimentRunner generates manifest automatically
- [ ] Manifest saved to `results/manifests/{experiment_name}/manifest.json`
- [ ] Manifest file is valid JSON

### 4.3 Reproducibility Verification
- [ ] Reproducibility script exists: `scripts/verify_reproducibility.py`
- [ ] Script shows help: `python scripts/verify_reproducibility.py --help`
- [ ] Can verify reproducibility of a manifest (if manifest exists)

---

## Test 5: CLI Completion & Integration (04-05)

### 5.1 CLI Commands
- [ ] `bsa --help` shows all commands (generate, run, analyze, reproduce, sweep)
- [ ] `bsa generate --help` works
- [ ] `bsa run --help` works
- [ ] `bsa analyze --help` works
- [ ] `bsa reproduce --help` works
- [ ] `bsa sweep --help` works

### 5.2 Full Reproduction Pipeline
- [ ] `bsa reproduce --small` runs without errors (may take time)
- [ ] Pipeline completes all steps (generate, run, analyze message)
- [ ] Results are saved correctly
- [ ] Manifests are generated

### 5.3 Error Handling
- [ ] Missing config files show clear error messages
- [ ] Invalid configs show clear error messages
- [ ] Progress output is clear and informative

---

## Test 6: End-to-End Integration

### 6.1 Small Dataset Test
- [ ] Can run full pipeline with `--small` flag
- [ ] Small dataset uses reduced parameters (10 episodes, 2 runs)
- [ ] Results are generated correctly

### 6.2 Results Structure
- [ ] Results directory structure is correct (`results/metrics/`, `results/manifests/`)
- [ ] Parquet files can be loaded
- [ ] JSON manifests can be loaded
- [ ] Results contain expected columns/metrics

---

## Overall Assessment

- [ ] All critical functionality works
- [ ] Error handling is appropriate
- [ ] Output is clear and informative
- [ ] Documentation is sufficient

## Issues Found
(List any issues encountered during testing)
