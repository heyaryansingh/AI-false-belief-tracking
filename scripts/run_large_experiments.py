#!/usr/bin/env python3
"""Run large-scale experiments for research data collection.

This script executes comprehensive experiments with:
- 5000-10000 episodes
- 50+ runs per configuration
- All models and conditions
- Real simulation results (not samples)
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
from bsa.experiments.run_experiment import generate_episodes, run_experiments
from bsa.analysis.aggregate import analyze_results


def main():
    """Run large-scale experiments."""
    print("=" * 70)
    print("Large-Scale Research Experiment Execution")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    config_path = Path("configs/experiments/exp_large_scale.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please ensure configs/experiments/exp_large_scale.yaml exists")
        return 1
    
    config = load_config(config_path)
    
    # Extract configurations
    gen_config = config.get("generator", {})
    exp_config = config.get("experiment", {})
    analysis_config = config.get("analysis", {})
    
    # Print experiment plan
    num_episodes = gen_config.get("num_episodes", 10000)
    num_runs = exp_config.get("num_runs", 50)
    models = exp_config.get("models", [])
    conditions = exp_config.get("conditions", [])
    
    total_runs = len(models) * len(conditions) * num_runs
    
    print("Experiment Plan:")
    print(f"  Episodes to generate: {num_episodes:,}")
    print(f"  Models: {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Runs per configuration: {num_runs}")
    print(f"  Total experiment runs: {total_runs:,}")
    print(f"  Total episodes to evaluate: {num_episodes * total_runs:,}")
    print()
    
    # Confirm execution
    response = input("Proceed with large-scale experiment execution? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Aborted.")
        return 0
    
    start_time = time.time()
    
    # Step 1: Generate episodes
    print("\n" + "=" * 70)
    print("[1/3] Generating Episodes")
    print("=" * 70)
    episode_start = time.time()
    
    try:
        episodes = generate_episodes(gen_config)
        episode_time = time.time() - episode_start
        print(f"\n✓ Generated {len(episodes):,} episodes in {episode_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n✗ Episode generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: Run experiments
    print("\n" + "=" * 70)
    print("[2/3] Running Experiments")
    print("=" * 70)
    exp_start = time.time()
    
    try:
        # Create experiment config dict
        experiment_config = {
            "experiment": exp_config
        }
        
        results = run_experiments(experiment_config)
        exp_time = time.time() - exp_start
        print(f"\n✓ Experiments completed in {exp_time/60:.1f} minutes")
        print(f"  Results saved to: {results.get('output_dir', 'results/metrics')}")
    except Exception as e:
        print(f"\n✗ Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Analyze results
    print("\n" + "=" * 70)
    print("[3/3] Analyzing Results")
    print("=" * 70)
    analysis_start = time.time()
    
    try:
        analyze_results(analysis_config)
        analysis_time = time.time() - analysis_start
        print(f"\n✓ Analysis completed in {analysis_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Experiment Execution Complete")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Results:")
    print(f"  Episodes: data/episodes/large_scale/")
    print(f"  Metrics: results/metrics/large_scale_research/")
    print(f"  Analysis: results/analysis/large_scale/")
    print(f"  Figures: results/analysis/large_scale/figures/")
    print(f"  Tables: results/analysis/large_scale/tables/")
    print()
    
    # Save execution log
    log_path = Path("results/execution_logs")
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_episodes": num_episodes,
            "num_runs": num_runs,
            "models": models,
            "conditions": conditions,
            "total_runs": total_runs,
        },
        "timing": {
            "episode_generation_minutes": episode_time / 60,
            "experiment_execution_minutes": exp_time / 60,
            "analysis_minutes": analysis_time / 60,
            "total_hours": total_time / 3600,
        },
        "results": {
            "episodes_generated": len(episodes),
            "output_dir": str(results.get("output_dir", "")),
        }
    }
    
    log_file.write_text(json.dumps(log_data, indent=2))
    print(f"Execution log saved to: {log_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
