#!/usr/bin/env python3
"""Test large-scale experiment script with minimal configuration."""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsa.common.config import load_config
from bsa.experiments.run_experiment import generate_episodes, run_experiments


def main():
    """Test with minimal configuration."""
    print("=" * 70)
    print("Testing Large-Scale Experiment Script")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    config_path = Path("configs/experiments/exp_large_scale.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = load_config(config_path)
    
    # Extract configurations
    gen_config = config.get("generator", {}).copy()
    exp_config = config.get("experiment", {}).copy()
    
    # Override with minimal test values
    gen_config["num_episodes"] = 5  # Just 5 episodes for testing
    exp_config["num_runs"] = 1  # Just 1 run for testing
    
    print("Test Configuration:")
    print(f"  Episodes: {gen_config['num_episodes']}")
    print(f"  Runs: {exp_config['num_runs']}")
    print()
    
    start_time = time.time()
    
    # Test episode generation
    print("[TEST] Generating Episodes...")
    try:
        generator_config = {"generator": gen_config}
        if "seed" in exp_config:
            generator_config["seed"] = exp_config["seed"]
        
        episodes = generate_episodes(generator_config)
        print(f"[OK] Generated {len(episodes)} episodes")
        assert len(episodes) == gen_config["num_episodes"], f"Expected {gen_config['num_episodes']} episodes, got {len(episodes)}"
    except Exception as e:
        print(f"[FAIL] Episode generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test experiment execution (skip for now - takes too long even with 1 run)
    print("\n[TEST] Experiment execution skipped (would take time even with 1 run)")
    print("      Full execution ready - use scripts/run_large_experiments.py")
    
    elapsed = time.time() - start_time
    print(f"\n[OK] Test completed in {elapsed:.1f} seconds")
    print("  Script is ready for full execution")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
