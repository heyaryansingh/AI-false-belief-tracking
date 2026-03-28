#!/usr/bin/env python3
"""Run ablation study for particle counts.

Tests BeliefSensitiveHelper with varying particle counts to prove the "Blind Spot"
persists regardless of compute.
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
from bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper

def main():
    parser = argparse.ArgumentParser(description="Run particle count ablations")
    parser.add_argument("--episodes", "-n", type=int, default=50,
                       help="Number of episodes per config")
    parser.add_argument("--runs", "-r", type=int, default=2,
                       help="Number of runs per config")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6 Rigor: Particle Count Ablation Study")
    print("=" * 70)
    
    particle_counts = [50, 100, 500, 1000]
    drift_prob = 1.0  # Only test in the False Belief condition (where Blind Spot exists)
    condition = "false_belief"
    
    all_results = []
    start_time = time.time()
    
    for particles in particle_counts:
        print(f"\n[Particles={particles}] (drift={drift_prob})")
        
        for run_idx in range(args.runs):
            base_seed = 30000 + particles + run_idx * 1000 
            
            for ep_idx in range(args.episodes):
                seed = base_seed + ep_idx
                env = GridHouseEnvironment(seed=seed)
                gen = GridHouseEpisodeGenerator(env, seed=seed, drift_probability=drift_prob)
                
                try:
                    episode = gen.generate_episode()
                except Exception as e:
                    print(f"  Warning: Gen failed: {e}")
                    continue

                # Instantiate helper with specific particle count
                helper = BeliefSensitiveHelper(seed=seed, num_particles=particles)
                evaluator = EpisodeEvaluator()
                
                try:
                    metrics = evaluator.evaluate_episode(episode, helper)
                except Exception as e:
                    print(f"  Warning: Eval failed: {e}")
                    continue
                
                result = {
                    "model": "belief_pf",
                    "particles": particles,
                    "condition": condition,
                    "run": run_idx,
                    "episode_id": episode.episode_id,
                    "auroc": metrics.get("false_belief_detection_auroc"),
                    "max_detection_score": metrics.get("max_detection_score", 0.0), # Added this metric
                    "efficiency": metrics.get("task_efficiency"),
                }
                all_results.append(result)
            
            print(f"  Run {run_idx+1}/{args.runs} complete.", end="\r")
        print()

    # Save results
    output_dir = Path("results/metrics/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_parquet(output_dir / "results.parquet")
    print(f"\nAblation results saved to {output_dir}")
    
    # Quick summary
    print("\nAbation Summary (AUROC):")
    print(df.groupby("particles")["auroc"].mean())

if __name__ == "__main__":
    main()
