#!/usr/bin/env python3
"""Run baselines to get max_detection_score for ROC curves."""

import sys
from pathlib import Path
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.envs.gridhouse.env import GridHouseEnvironment
from bsa.envs.gridhouse.episode_generator import GridHouseEpisodeGenerator
from bsa.experiments.evaluator import EpisodeEvaluator
from bsa.agents.helper.reactive import ReactiveHelper
from bsa.agents.helper.goal_only import GoalOnlyHelper

def main():
    print("Generating Baseline Data for ROC Curves...")
    
    models = ["reactive", "goal_only"]
    conditions = ["control", "false_belief"] # Need + and - classes
    episodes = 50
    runs = 2
    
    all_results = []
    
    for model_name in models:
        print(f"Running {model_name}...")
        for condition in conditions:
            drift_prob = 1.0 if condition == "false_belief" else 0.0
            
            for run in range(runs):
                base_seed = 40000 + run * 1000
                for ep in range(episodes):
                    seed = base_seed + ep
                    env = GridHouseEnvironment(seed=seed)
                    gen = GridHouseEpisodeGenerator(env, seed=seed, drift_probability=drift_prob)
                    try:
                        episode = gen.generate_episode()
                    except: continue
                    
                    if model_name == "reactive":
                        helper = ReactiveHelper(seed=seed)
                    else:
                        helper = GoalOnlyHelper(seed=seed)
                        
                    evaluator = EpisodeEvaluator()
                    metrics = evaluator.evaluate_episode(episode, helper)
                    
                    all_results.append({
                        "model": model_name,
                        "condition": condition,
                        "max_detection_score": metrics.get("max_detection_score", 0.0),
                        "efficiency": metrics.get("task_efficiency")
                    })
                    
    df = pd.DataFrame(all_results)
    output_dir = Path("results/metrics/baselines_roc")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "results.parquet")
    print("Baseline ROC data saved.")

if __name__ == "__main__":
    main()
