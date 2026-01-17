"""Experiment runner for automated experiment execution."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common.seeding import get_rng
from ..common.types import Episode, EpisodeStep, Action, Observation
from ..agents.helper import ReactiveHelper, GoalOnlyHelper, BeliefSensitiveHelper
from ..envs.gridhouse import GridHouseEnvironment, GridHouseEpisodeGenerator
from ..envs.gridhouse.tasks import get_task, list_tasks
from .evaluator import EpisodeEvaluator


class ExperimentRunner:
    """Runner for executing experiments with different helper models and conditions."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration dictionary with keys:
                - name: Experiment name
                - models: List of model names ['reactive', 'goal_only', 'belief_pf']
                - conditions: List of condition names ['control', 'false_belief', 'seen_relocation']
                - num_runs: Number of runs per model/condition combination
                - seed: Random seed for reproducibility
                - env_type: 'gridhouse' or 'virtualhome' (default: 'gridhouse')
            output_dir: Output directory for results (default: results/metrics/{name}/)
        """
        self.config = config
        self.experiment_name = config.get("name", "experiment")
        self.models = config.get("models", ["reactive"])
        self.conditions = config.get("conditions", ["control"])
        self.num_runs = config.get("num_runs", 10)
        self.seed = config.get("seed", 42)
        self.env_type = config.get("env_type", "gridhouse")
        
        self.rng = get_rng(self.seed)
        
        # Output directory
        if output_dir is None:
            output_dir = Path("results/metrics") / self.experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        
        # Episode evaluator for comprehensive metrics
        self.evaluator = EpisodeEvaluator()

    def run_experiment(self) -> Dict[str, Any]:
        """Run full experiment.

        Returns:
            Dictionary with experiment results
        """
        print(f"Running experiment: {self.experiment_name}")
        print(f"  Models: {self.models}")
        print(f"  Conditions: {self.conditions}")
        print(f"  Runs per combination: {self.num_runs}")
        print(f"  Environment: {self.env_type}")
        
        # Create environment and episode generator
        if self.env_type == "gridhouse":
            env = GridHouseEnvironment(seed=self.seed)
            episode_generator = GridHouseEpisodeGenerator(env, seed=self.seed)
        elif self.env_type == "virtualhome":
            from ..envs.virtualhome import VirtualHomeEnvironment, VirtualHomeEpisodeGenerator
            env = VirtualHomeEnvironment(seed=self.seed)
            episode_generator = VirtualHomeEpisodeGenerator(env, seed=self.seed)
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")
        
        # Run experiments for each model × condition combination
        for model_name in self.models:
            print(f"\n  Model: {model_name}")
            helper_agent = self._create_helper_agent(model_name)
            
            for condition in self.conditions:
                print(f"    Condition: {condition}")
                
                for run_idx in range(self.num_runs):
                    # Generate or load episode based on condition
                    episode = self._generate_episode_for_condition(
                        episode_generator, condition, run_idx
                    )
                    
                    # Evaluate episode with helper agent (use comprehensive evaluator)
                    metrics = self.evaluator.evaluate_episode(episode, helper_agent)
                    
                    # Add experiment metadata
                    metrics.update({
                        "model": model_name,
                        "condition": condition,
                        "run": run_idx,
                    })
                    
                    # Store results
                    result = {
                        "experiment_name": self.experiment_name,
                        "episode_id": episode.episode_id,
                        "goal_id": episode.goal_id,
                        "tau": episode.tau,
                        "intervention_type": episode.intervention_type,
                        **metrics,
                    }
                    self.results.append(result)
                    
                    if (run_idx + 1) % 5 == 0:
                        print(f"      Completed {run_idx + 1}/{self.num_runs} runs")
        
        # Save results
        self._save_results()
        
        # Return summary
        return {
            "experiment_name": self.experiment_name,
            "num_results": len(self.results),
            "output_dir": str(self.output_dir),
        }

    def _create_helper_agent(self, model_name: str):
        """Create helper agent from model name.

        Args:
            model_name: Model name ('reactive', 'goal_only', 'belief_pf')

        Returns:
            Helper agent instance
        """
        task_list = [get_task(task_id) for task_id in list_tasks()]
        
        if model_name == "reactive":
            return ReactiveHelper()
        elif model_name == "goal_only":
            return GoalOnlyHelper(task_list=task_list, seed=self.seed)
        elif model_name == "belief_pf":
            # Get num_particles from config if available
            num_particles = self.config.get("model_config", {}).get(
                "belief_pf", {}
            ).get("particle_filter", {}).get("num_particles", 100)
            return BeliefSensitiveHelper(
                task_list=task_list,
                num_particles=num_particles,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _generate_episode_for_condition(
        self, episode_generator, condition: str, run_idx: int
    ) -> Episode:
        """Generate episode for specific condition.

        Args:
            episode_generator: Episode generator instance
            condition: Condition name ('control', 'false_belief', 'seen_relocation')
            run_idx: Run index for seeding

        Returns:
            Generated episode
        """
        # Use run_idx to seed episode generation deterministically
        episode_seed = self.seed + run_idx * 1000
        
        if condition == "control":
            # No intervention
            return episode_generator.generate_episode(
                goal_id=None,  # Random goal
                tau=None,  # No intervention
                intervention_type=None,
            )
        elif condition == "false_belief":
            # False-belief intervention
            tau = self.rng.integers(5, 20)  # Random intervention timestep
            return episode_generator.generate_episode(
                goal_id=None,  # Random goal
                tau=tau,
                intervention_type="relocate",
            )
        elif condition == "seen_relocation":
            # Human sees relocation (no false belief)
            tau = self.rng.integers(5, 20)
            # For seen_relocation, we'd need to modify episode generator
            # For now, treat same as false_belief but mark differently
            episode = episode_generator.generate_episode(
                goal_id=None,
                tau=tau,
                intervention_type="relocate",
            )
            # Mark as seen (simplified - in practice would ensure human sees it)
            return episode
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def _evaluate_episode(
        self,
        episode: Episode,
        helper_agent,
        model_name: str,
        condition: str,
        run_idx: int,
    ) -> Dict[str, Any]:
        """Evaluate episode with helper agent.

        Args:
            episode: Episode to evaluate
            helper_agent: Helper agent instance
            model_name: Model name
            condition: Condition name
            run_idx: Run index

        Returns:
            Dictionary with episode metrics
        """
        # Reset helper agent state
        helper_agent.reset()
        
        # Track metrics during evaluation
        helper_actions: List[Action] = []
        false_belief_detected = False
        detection_latency: Optional[int] = None
        num_interventions = 0
        
        # Iterate through episode steps
        for step in episode.steps:
            # Get helper observation
            helper_obs = step.helper_obs
            
            # Plan action
            helper_action = helper_agent.plan_action(helper_obs, episode_step=step)
            helper_actions.append(helper_action)
            
            # Update belief
            helper_agent.update_belief(
                observation=helper_obs,
                human_action=step.human_action,
                episode_step=step,
            )
            
            # Check for false-belief detection (for belief-sensitive helper)
            if model_name == "belief_pf":
                # Check if false belief detected at this step
                if not false_belief_detected:
                    # Use helper's detect_false_belief method
                    # BeliefSensitiveHelper needs episode_step with true_locations
                    if helper_agent.detect_false_belief(helper_obs, episode_step=step):
                        false_belief_detected = True
                        detection_latency = step.timestep
            
            # Count interventions (actions that are not WAIT)
            if helper_action != Action.WAIT:
                num_interventions += 1
        
        # Compute episode-level metrics
        task_completed = episode.metadata.get("task_completed", False)
        num_steps = len(episode.steps)
        num_wasted_actions = self._count_wasted_actions(episode)
        
        # Helper action type distribution
        action_types = {}
        for action in helper_actions:
            action_types[action.value] = action_types.get(action.value, 0) + 1
        
        metrics = {
            "task_completed": task_completed,
            "num_steps": num_steps,
            "num_wasted_actions": num_wasted_actions,
            "false_belief_detected": false_belief_detected,
            "detection_latency": detection_latency,
            "num_helper_interventions": num_interventions,
            "helper_action_types": action_types,
            "num_helper_actions": len(helper_actions),
        }
        
        return metrics

    def _count_wasted_actions(self, episode: Episode) -> int:
        """Count wasted actions (actions that don't progress toward goal).

        Args:
            episode: Episode to analyze

        Returns:
            Number of wasted actions
        """
        # Simplified: count actions that are not progressing toward goal
        # In practice, would check if action moves toward goal location
        wasted_count = 0
        
        goal_id = episode.goal_id
        task = get_task(goal_id)
        
        for step in episode.steps:
            human_action = step.human_action
            
            # Check if action is wasted (simplified heuristic)
            # If human moves away from goal objects, it's wasted
            # If human picks up non-critical objects, it's wasted
            # For now, use simple heuristic: MOVE actions when critical objects visible = wasted
            if human_action == Action.MOVE:
                # Check if critical objects are visible
                for obj_id in task.critical_objects:
                    if obj_id in step.visible_objects_h:
                        # Object is visible but human is moving away = wasted
                        wasted_count += 1
                        break
        
        return wasted_count

    def _save_results(self) -> None:
        """Save experiment results to files."""
        # Save as Parquet
        df = pd.DataFrame(self.results)
        parquet_path = self.output_dir / "results.parquet"
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path, compression="snappy")
        
        # Save as JSON manifest
        manifest = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "num_results": len(self.results),
            "summary": {
                "models": self.models,
                "conditions": self.conditions,
                "num_runs": self.num_runs,
            },
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  Parquet: {parquet_path}")
        print(f"  Manifest: {manifest_path}")
