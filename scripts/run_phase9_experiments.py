#!/usr/bin/env python3
"""Run Phase 10 experiments with statistical strengthening.

# Fix: Added partial_false_belief condition for intermediate belief state (Phase 10)
# Fix: Seed logging for full reproducibility (Phase 10)
# Fix: Enhanced metrics with temporal tracking and bootstrap CIs (Phase 10)

This script runs experiments with:
1. Three conditions: control, partial_false_belief, false_belief
2. Bootstrap confidence intervals for AUROC
3. Temporal metrics (time-to-detection, false alarm rate)
4. Effect size calculations for model comparisons
5. Full seed logging for reproducibility
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import json
import argparse
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.envs.gridhouse.env import GridHouseEnvironment
from bsa.envs.gridhouse.episode_generator import GridHouseEpisodeGenerator
from bsa.experiments.evaluator import EpisodeEvaluator
from bsa.agents.helper.reactive import ReactiveHelper
from bsa.agents.helper.goal_only import GoalOnlyHelper
from bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper
from bsa.analysis.statistics import (
    compute_bootstrap_ci,
    effect_size,
    format_ci,
    format_p_value,
)


def get_helper(model_name: str, seed: int):
    """Create helper agent by name."""
    if model_name == "reactive":
        return ReactiveHelper(seed=seed)
    elif model_name == "goal_only":
        return GoalOnlyHelper(seed=seed)
    elif model_name == "belief_pf":
        return BeliefSensitiveHelper(seed=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 9 experiments")
    parser.add_argument("--episodes", "-n", type=int, default=100,
                       help="Number of episodes per condition")
    parser.add_argument("--runs", "-r", type=int, default=3,
                       help="Number of runs per model/condition")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal episodes")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 20
        args.runs = 2

    print("=" * 70)
    print("Phase 10: Statistical Strengthening Experiments")
    print("=" * 70)
    print(f"Episodes per condition: {args.episodes}")
    print(f"Runs per config: {args.runs}")
    print()

    models = ["reactive", "goal_only", "belief_pf"]
    
    # Fix: Added partial_false_belief condition for intermediate belief state (Phase 10)
    # Maps condition name to drift_probability
    conditions = {
        "control": 0.0,
        "partial_false_belief": 0.5,
        "false_belief": 1.0,
    }

    all_results = []
    seed_manifest = []  # Fix: Seed logging for full reproducibility (Phase 10)
    start_time = time.time()

    for condition, drift_prob in conditions.items():
        print(f"\n[{condition.upper()}] (drift_probability={drift_prob})")

        for run_idx in range(args.runs):
            base_seed = 10000 + run_idx * 1000

            for ep_idx in range(args.episodes):
                seed = base_seed + ep_idx
                env = GridHouseEnvironment(seed=seed)
                gen = GridHouseEpisodeGenerator(
                    env,
                    seed=seed,
                    drift_probability=drift_prob  # Fix: Use condition-specific drift probability
                )
                
                # Fix: Log seed for reproducibility (Phase 10)
                seed_manifest.append({
                    "condition": condition,
                    "run": run_idx,
                    "episode_idx": ep_idx,
                    "seed": seed,
                    "drift_probability": drift_prob,
                })

                try:
                    episode = gen.generate_episode()
                except Exception as e:
                    print(f"  Warning: Episode generation failed: {e}")
                    continue

                evaluator = EpisodeEvaluator()

                for model_name in models:
                    helper = get_helper(model_name, seed)

                    try:
                        metrics = evaluator.evaluate_episode(episode, helper)
                    except Exception as e:
                        print(f"  Warning: Evaluation failed for {model_name}: {e}")
                        continue

                    # Fix: Extended result dictionary with temporal metrics (Phase 10)
                    result = {
                        "model": model_name,
                        "condition": condition,
                        "run": run_idx,
                        "episode_id": episode.episode_id,
                        "seed": seed,  # Fix: Include seed for reproducibility
                        # Detection metrics
                        "auroc": metrics.get("false_belief_detection_auroc"),
                        "latency": metrics.get("false_belief_detection_latency"),
                        "fpr": metrics.get("false_belief_detection_fpr"),
                        # Fix: Added temporal metrics (Phase 10)
                        "time_to_detection": metrics.get("time_to_detection"),
                        "false_alarm_rate": metrics.get("false_alarm_rate"),
                        "false_belief_onset": metrics.get("false_belief_onset_timestep"),
                        # Task metrics
                        "task_completed": metrics.get("task_completed"),
                        "efficiency": metrics.get("task_efficiency"),
                        "wasted_actions": metrics.get("num_wasted_actions"),
                        # Fix: Added detailed wasted action breakdown (Phase 10)
                        "wasted_move_visible": metrics.get("wasted_move_when_visible"),
                        "wasted_failed_pickup": metrics.get("wasted_failed_pickup"),
                        "wasted_backtracking": metrics.get("wasted_backtracking"),
                        # Intervention metrics
                        "interventions": metrics.get("num_interventions"),
                        "precision": metrics.get("intervention_precision"),
                        "recall": metrics.get("intervention_recall"),
                        "f1": metrics.get("intervention_f1"),
                        "helper_wait_count": metrics.get("helper_wait_count"),
                        "helper_intervention_count": metrics.get("helper_intervention_count"),
                    }
                    all_results.append(result)

            # Progress
            total = len(conditions) * args.runs * args.episodes * len(models)
            done = len(all_results)
            print(f"  Run {run_idx + 1}/{args.runs}: {done} evaluations", end="\r")

        print()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Print summary with bootstrap CIs
    print("\n" + "=" * 70)
    print("Results Summary (Phase 10: Statistical Strengthening)")
    print("=" * 70)

    # Fix: Display statistics with CIs instead of just mean ± SD (Phase 10)
    print("\n** AUROC (False Belief Detection) with Bootstrap 95% CI **")
    for model in models:
        model_auroc = df[(df["model"] == model) & (df["auroc"].notna())]["auroc"].values
        if len(model_auroc) > 0:
            ci_result = compute_bootstrap_ci(model_auroc, n_bootstrap=1000)
            print(f"  {model}: {format_ci(ci_result['value'], ci_result['ci_lower'], ci_result['ci_upper'])}")
        else:
            print(f"  {model}: N/A (no valid AUROC values)")

    # Check for suspicious values
    print("\n** Data Validation Checks **")
    belief_auroc = df[(df["model"] == "belief_pf") & (df["auroc"].notna())]["auroc"]
    if len(belief_auroc) > 0:
        if belief_auroc.mean() >= 0.999:
            print("  WARNING: belief_pf AUROC still ~1.0 - data leakage may persist")
        else:
            print(f"  OK: belief_pf AUROC = {belief_auroc.mean():.3f} ± {belief_auroc.std():.3f}")

    # Check model differentiation with effect sizes
    print("\n** Model Comparisons (Effect Sizes) **")
    auroc_by_model = {}
    for model in models:
        values = df[(df["model"] == model) & (df["auroc"].notna())]["auroc"].values
        if len(values) > 0:
            auroc_by_model[model] = values
    
    if "belief_pf" in auroc_by_model and "goal_only" in auroc_by_model:
        d = effect_size(auroc_by_model["belief_pf"], auroc_by_model["goal_only"])
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"  Belief-Sensitive vs Goal-Only: d = {d:.3f} ({interp} effect)")
    
    if "belief_pf" in auroc_by_model and "reactive" in auroc_by_model:
        d = effect_size(auroc_by_model["belief_pf"], auroc_by_model["reactive"])
        interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"  Belief-Sensitive vs Reactive: d = {d:.3f} ({interp} effect)")

    # Condition comparison
    print("\n** Conditions Summary **")
    for condition in conditions.keys():
        cond_df = df[df["condition"] == condition]
        auroc_vals = cond_df[cond_df["auroc"].notna()]["auroc"]
        if len(auroc_vals) > 0:
            print(f"  {condition}: N={len(cond_df)}, AUROC mean={auroc_vals.mean():.3f}")
        else:
            print(f"  {condition}: N={len(cond_df)}, AUROC=N/A")

    # Task completion
    print("\n** Task Completion **")
    completion_rate = df.groupby("model")["task_completed"].mean() * 100
    print(completion_rate.to_string())
    if completion_rate.mean() == 0:
        print("  WARNING: No tasks completed - check task completion logic")

    # Efficiency by model
    print("\n** Efficiency by Model (with 95% CI) **")
    for model in models:
        model_eff = df[df["model"] == model]["efficiency"].dropna().values
        if len(model_eff) > 0:
            ci_result = compute_bootstrap_ci(model_eff, n_bootstrap=1000)
            print(f"  {model}: {format_ci(ci_result['value'], ci_result['ci_lower'], ci_result['ci_upper'])}")

    # Fix: Temporal metrics summary (Phase 10)
    print("\n** Temporal Metrics (Time-to-Detection) **")
    for model in models:
        model_ttd = df[(df["model"] == model) & (df["time_to_detection"].notna())]["time_to_detection"].values
        if len(model_ttd) > 0:
            ci_result = compute_bootstrap_ci(model_ttd, n_bootstrap=1000)
            print(f"  {model}: {format_ci(ci_result['value'], ci_result['ci_lower'], ci_result['ci_upper'])} timesteps")
        else:
            print(f"  {model}: N/A")

    # Save results
    output_dir = Path("results/metrics/phase10_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.parquet"
    df.to_parquet(output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Fix: Save seed manifest for reproducibility (Phase 10)
    manifest_path = output_dir / "seed_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": {
                "episodes": args.episodes,
                "runs": args.runs,
                "quick": args.quick,
            },
            "conditions": conditions,
            "models": models,
            "seeds": seed_manifest,
        }, f, indent=2)
    print(f"Seed manifest saved to: {manifest_path}")

    # Timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
