"""Experiment runner."""

from pathlib import Path
from typing import Dict, Any, Optional, List

from ..common.config import load_config
from ..common.seeding import get_rng
from .runner import ExperimentRunner
from ..envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
from ..envs.gridhouse.tasks import list_tasks


def generate_episodes(config: Dict[str, Any], output_dir: Optional[Path] = None, seed: Optional[int] = None) -> List:
    """Generate episodes from generator config.

    Args:
        config: Generator configuration dictionary with keys:
            - generator.env_config: Environment config path
            - generator.num_episodes: Number of episodes to generate
            - generator.tau_range: [min, max] intervention timestep range
            - generator.drift_probability: Probability of intervention
            - generator.occlusion_severity: Occlusion severity (0-1)
            - generator.output_format: 'parquet' or 'jsonl'
            - generator.output_dir: Output directory
            - tasks: List of task IDs to use
        output_dir: Override output directory from config
        seed: Override seed from config

    Returns:
        List of generated episodes
    """
    gen_config = config.get("generator", {})
    
    # Determine environment type
    env_config_path = gen_config.get("env_config", "")
    if "gridhouse" in env_config_path.lower() or not env_config_path:
        env_type = "gridhouse"
    elif "virtualhome" in env_config_path.lower():
        env_type = "virtualhome"
    else:
        env_type = "gridhouse"  # Default
    
    # Get parameters
    num_episodes = gen_config.get("num_episodes", 100)
    tau_range = tuple(gen_config.get("tau_range", [5, 20]))
    drift_probability = gen_config.get("drift_probability", 0.5)
    occlusion_severity = gen_config.get("occlusion_severity", 0.5)
    output_format = gen_config.get("output_format", "parquet")
    
    if output_dir is None:
        output_dir = Path(gen_config.get("output_dir", "data/episodes"))
    else:
        output_dir = Path(output_dir)
    
    if seed is None:
        seed = config.get("seed", 42)
    
    # Get task distribution
    task_list = config.get("tasks", list_tasks())
    
    print(f"Generating {num_episodes} episodes")
    print(f"  Environment: {env_type}")
    print(f"  Output: {output_dir}")
    print(f"  Format: {output_format}")
    print(f"  Tasks: {task_list}")
    
    # Create environment and generator
    rng = get_rng(seed)
    
    if env_type == "gridhouse":
        env = GridHouseEnvironment(seed=seed)
        generator = GridHouseEpisodeGenerator(
            env,
            seed=seed,
            tau_range=tau_range,
            drift_probability=drift_probability,
            occlusion_severity=occlusion_severity,
        )
    elif env_type == "virtualhome":
        from ..envs.virtualhome import VirtualHomeEnvironment, VirtualHomeEpisodeGenerator
        env = VirtualHomeEnvironment(seed=seed)
        generator = VirtualHomeEpisodeGenerator(
            env,
            seed=seed,
            tau_range=tau_range,
            drift_probability=drift_probability,
            occlusion_severity=occlusion_severity,
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    # Generate episodes
    episodes = generator.generate_episodes(
        num_episodes=num_episodes,
        output_dir=output_dir,
        format=output_format,
        goal_distribution=task_list,
    )
    
    print(f"\nGenerated {len(episodes)} episodes")
    return episodes


def run_experiments(config: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    """Run experiments from experiment config.

    Args:
        config: Experiment configuration dictionary (see ExperimentRunner.__init__)
        output_dir: Override output directory from config
    """
    # Create experiment runner
    if output_dir:
        config["output_dir"] = str(output_dir)
    
    runner = ExperimentRunner(config, output_dir=output_dir)
    
    # Run experiments
    results = runner.run_experiment()
    
    print(f"\nExperiment complete: {results['experiment_name']}")
    print(f"  Results: {results['num_results']} runs")
    print(f"  Output: {results['output_dir']}")


def reproduce(small: bool = False) -> None:
    """Full reproduction pipeline.

    Args:
        small: If True, use small dataset for CI/testing
    """
    print("=" * 70)
    print("Full Reproduction Pipeline")
    print("=" * 70)
    
    # Load configs
    generator_config_path = Path("configs/generator/default.yaml")
    experiment_config_path = Path("configs/experiments/exp_main.yaml")
    
    if not generator_config_path.exists():
        print(f"Warning: Generator config not found: {generator_config_path}")
        return
    
    if not experiment_config_path.exists():
        print(f"Warning: Experiment config not found: {experiment_config_path}")
        return
    
    # Load configs
    generator_config = load_config(generator_config_path)
    experiment_config = load_config(experiment_config_path)
    
    # Adjust for small dataset
    if small:
        print("Using small dataset for CI/testing")
        generator_config["generator"]["num_episodes"] = 10
        experiment_config["experiment"]["num_runs"] = 2
    
    # Step 1: Generate episodes
    print("\n[1/3] Generating episodes...")
    generate_episodes(generator_config)
    
    # Step 2: Run experiments
    print("\n[2/3] Running experiments...")
    run_experiments(experiment_config)
    
    # Step 3: Analyze (will be implemented in Phase 5)
    print("\n[3/3] Analysis...")
    print("  Analysis will be implemented in Phase 5")
    
    print("\n" + "=" * 70)
    print("Reproduction complete!")
    print("=" * 70)
