"""Massive parallel runner for Phase 7 Expansion.

Target: 27,000 Episodes.
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import json
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.envs.gridhouse.env import GridHouseEnvironment
from bsa.envs.gridhouse.episode_generator import GridHouseEpisodeGenerator
from bsa.envs.virtualhome.env import VirtualHomeEnvironment
from bsa.envs.virtualhome.generator import VirtualHomeEpisodeGenerator

from bsa.experiments.evaluator import EpisodeEvaluator

from bsa.agents.helper.reactive import ReactiveHelper
from bsa.agents.helper.goal_only import GoalOnlyHelper
from bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper
from bsa.agents.helper.active import ActiveVerificationHelper
from bsa.agents.helper.communication import CommunicationHelper
from bsa.agents.helper.oracle import OracleHelper

def run_single_episode(seed, condition, model_name, env_type):
    """Run a single episode and return metrics."""
    drift_prob = 0.0
    if condition == "drift": drift_prob = 0.5
    elif condition == "blind_spot": drift_prob = 1.0
    
    # Environment Setup
    if env_type == "gridhouse":
        env = GridHouseEnvironment(seed=seed)
        gen = GridHouseEpisodeGenerator(env, seed=seed, drift_probability=drift_prob)
    else:
        env = VirtualHomeEnvironment(seed=seed, objects=["keys", "phone", "book"])
        gen = VirtualHomeEpisodeGenerator(env, seed=seed, drift_probability=drift_prob)
        
    # Helper Setup
    if model_name == "reactive": helper = ReactiveHelper(seed=seed)
    elif model_name == "belief_pf": helper = BeliefSensitiveHelper(seed=seed, num_particles=100)
    elif model_name == "goal_proxy": helper = GoalOnlyHelper(seed=seed)
    elif model_name == "active": helper = ActiveVerificationHelper(seed=seed)
    elif model_name == "communication": helper = CommunicationHelper(seed=seed)
    elif model_name == "oracle": helper = OracleHelper(seed=seed)
    else: return None
    
    # Generate Episode (Closed Loop for VH, Standard for GH)
    try:
        if env_type == "virtualhome":
            episode = gen.generate_episode(helper_agent=helper)
        else:
            # GridHouse uses standard generation (offline evaluation)
            episode = gen.generate_episode()
    except Exception as e:
        return None
    
    # Evaluate (Reset helper to generate fresh detection scores on the recorded trace)
    helper.reset() # Important!
    evaluator = EpisodeEvaluator()
    try:
        metrics = evaluator.evaluate_episode(episode, helper)
    except Exception as e:
        return None
        
    return {
        "env": env_type,
        "condition": condition,
        "model": model_name,
        "seed": seed,
        "auroc": metrics.get("false_belief_detection_auroc"),
        "max_detection_score": metrics.get("max_detection_score", 0.0),
        "efficiency": metrics.get("task_efficiency"),
        "latency": metrics.get("false_belief_detection_latency"),
        "time_to_detection": metrics.get("time_to_detection"),
        "fpr": metrics.get("false_belief_detection_fpr"),
        "net_benefit": (metrics.get("num_wasted_actions", 0)*1.0) - (metrics.get("num_interventions", 0)*0.5) # Heuristic
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per config")
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs")
    args = parser.parse_args()
    
    print(f"Starting Massive Expansion Run ({args.episodes} eps/config)...")
    
    envs = ["gridhouse", "virtualhome"]
    conditions = ["control", "drift", "blind_spot"]
    models = ["reactive", "belief_pf", "goal_proxy", "active", "communication", "oracle"]
    
    tasks = []
    base_seed = 50000
    
    # Generate Task List
    count = 0
    for env in envs:
        for cond in conditions:
            for model in models:
                for i in range(args.episodes):
                    tasks.append((base_seed + count, cond, model, env))
                    count += 1
                    
    print(f"Total Tasks: {len(tasks)}")
    
    # Run Parallel
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_single_episode)(seed, cond, model, env) 
        for seed, cond, model, env in tasks
    )
    
    # Filter Nones
    results = [r for r in results if r is not None]
    
    # Save
    df = pd.DataFrame(results)
    output_dir = Path("results/metrics/expansion_phase7")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "results.parquet")
    print(f"Saved {len(df)} results to {output_dir}")

if __name__ == "__main__":
    main()
