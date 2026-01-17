# Phase 4 Plan 1: Experiment Runner Core Summary

**Implemented core experiment runner for automated experiment execution**

## Accomplishments

- Created ExperimentRunner class for running experiments
- Integrated with helper agents (reactive, goal-only, belief-sensitive)
- Implemented episode evaluation logic
- Results saving to results/metrics/
- CLI integration working

## Files Created/Modified

- `src/bsa/experiments/runner.py` - ExperimentRunner class
- `src/bsa/experiments/run_experiment.py` - Updated with run_experiments() and generate_episodes()
- `src/bsa/experiments/__init__.py` - Export ExperimentRunner

## Decisions Made

- Experiment structure: models × conditions × runs
- Helper agent instantiation from config (model name → helper class)
- Episode evaluation approach (step-by-step simulation with helper agent)
- Metrics to track:
  - Task completion, num steps
  - Wasted actions (simplified heuristic)
  - False-belief detection (for belief-sensitive helper)
  - Detection latency
  - Helper interventions and action types
- Results storage format (Parquet + JSON manifest)
- Condition handling: control (no intervention), false_belief (intervention), seen_relocation (simplified)

## Issues Encountered

- Helper agent detect_false_belief() signature requires episode_step parameter - fixed
- Wasted action counting uses simplified heuristic (can be enhanced later)
- Seen_relocation condition simplified (would need episode generator modification for full implementation)

## Next Step

Ready for 04-02-PLAN.md (Episode Evaluator enhancements and detailed metrics)
