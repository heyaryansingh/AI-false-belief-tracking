"""Comprehensive episode evaluator with detailed metrics computation."""

from typing import Dict, List, Any, Optional
import numpy as np

from ..common.types import Episode, EpisodeStep, Action, Observation, ObjectLocation
from ..agents.helper.base import HelperAgent
from ..envs.gridhouse.tasks import get_task


class EpisodeEvaluator:
    """Comprehensive episode evaluator with detailed metrics computation."""

    def evaluate_episode(
        self,
        episode: Episode,
        helper_agent: HelperAgent,
    ) -> Dict[str, Any]:
        """Evaluate episode with helper agent and compute comprehensive metrics.

        Args:
            episode: Episode to evaluate
            helper_agent: Helper agent instance

        Returns:
            Dictionary with comprehensive metrics
        """
        # Reset helper agent state
        helper_agent.reset()
        
        # Track step-by-step data for metrics computation
        step_data: List[Dict[str, Any]] = []
        helper_actions: List[Action] = []
        false_belief_predictions: List[bool] = []
        false_belief_ground_truth: List[bool] = []
        
        # Iterate through episode steps
        for step in episode.steps:
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
            
            # Track false-belief detection
            false_belief_predicted = helper_agent.detect_false_belief(helper_obs, episode_step=step)
            false_belief_predictions.append(false_belief_predicted)
            
            # Ground truth: check if false belief actually exists
            false_belief_exists = self._check_false_belief_exists(step)
            false_belief_ground_truth.append(false_belief_exists)
            
            # Store step data
            step_data.append({
                "timestep": step.timestep,
                "helper_action": helper_action,
                "false_belief_predicted": false_belief_predicted,
                "false_belief_exists": false_belief_exists,
                "belief_state": helper_agent.get_belief_state(),
            })
        
        # Compute comprehensive metrics
        metrics = {}
        
        # False-belief detection metrics
        metrics.update(self._compute_false_belief_detection_metrics(
            false_belief_predictions, false_belief_ground_truth, step_data
        ))
        
        # Belief tracking metrics
        metrics.update(self._compute_belief_tracking_metrics(step_data, episode))
        
        # Task performance metrics
        metrics.update(self._compute_task_performance_metrics(episode, helper_actions))
        
        # Intervention metrics
        metrics.update(self._compute_intervention_metrics(episode, helper_actions, step_data))
        
        return metrics

    def _check_false_belief_exists(self, step: EpisodeStep) -> bool:
        """Check if false belief actually exists in this step.

        Args:
            step: Episode step

        Returns:
            True if false belief exists (human belief != true location for any object)
        """
        for obj_id, true_loc in step.true_object_locations.items():
            if obj_id in step.human_belief_object_locations:
                human_belief_loc = step.human_belief_object_locations[obj_id]
                if true_loc.room_id != human_belief_loc.room_id:
                    return True  # False belief exists
        return False

    def _compute_false_belief_detection_metrics(
        self,
        predictions: List[bool],
        ground_truth: List[bool],
        step_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute false-belief detection metrics (AUROC, detection latency, FPR).

        Args:
            predictions: List of false-belief predictions (one per step)
            ground_truth: List of ground truth false-belief existence (one per step)
            step_data: Step-by-step data

        Returns:
            Dictionary with detection metrics
        """
        if not any(ground_truth):
            # No false beliefs in episode
            return {
                "false_belief_detection_auroc": None,
                "false_belief_detection_latency": None,
                "false_belief_detection_fpr": None,
                "false_belief_detected": False,
            }
        
        # Compute detection latency (first timestep when false belief detected)
        detection_latency = None
        first_false_belief_timestep = None
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            if gt and first_false_belief_timestep is None:
                first_false_belief_timestep = step_data[i]["timestep"]
            if pred and detection_latency is None and first_false_belief_timestep is not None:
                detection_latency = step_data[i]["timestep"] - first_false_belief_timestep
                break
        
        # Compute AUROC (simplified - using predictions as scores)
        # In practice, would use probability scores from belief inference
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(ground_truth)) > 1:  # Need both classes
                # Convert boolean predictions to scores (1.0 if True, 0.0 if False)
                scores = [1.0 if p else 0.0 for p in predictions]
                auroc = roc_auc_score(ground_truth, scores)
            else:
                auroc = None
        except ImportError:
            # sklearn not available - compute manually
            auroc = None
        
        # Compute false positive rate
        false_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
        true_negatives = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        
        return {
            "false_belief_detection_auroc": auroc,
            "false_belief_detection_latency": detection_latency,
            "false_belief_detection_fpr": fpr,
            "false_belief_detected": any(predictions),
        }

    def _compute_belief_tracking_metrics(
        self,
        step_data: List[Dict[str, Any]],
        episode: Episode,
    ) -> Dict[str, Any]:
        """Compute belief tracking metrics (accuracy, cross-entropy, Brier score).

        Args:
            step_data: Step-by-step data
            episode: Episode being evaluated

        Returns:
            Dictionary with belief tracking metrics
        """
        # Get goal inference accuracy
        goal_id = episode.goal_id
        goal_accuracy = None
        
        # Check final goal distribution
        if step_data:
            final_belief_state = step_data[-1].get("belief_state")
            if final_belief_state and "goal_distribution" in final_belief_state:
                goal_dist = final_belief_state["goal_distribution"]
                if goal_id in goal_dist:
                    # Check if goal has highest probability
                    max_prob_goal = max(goal_dist.items(), key=lambda x: x[1])[0]
                    goal_accuracy = 1.0 if max_prob_goal == goal_id else 0.0
        
        # Cross-entropy and Brier score would require probability distributions
        # For now, return simplified metrics
        
        return {
            "goal_inference_accuracy": goal_accuracy,
            "belief_tracking_cross_entropy": None,  # Would need full distributions
            "belief_tracking_brier_score": None,  # Would need full distributions
        }

    def _compute_task_performance_metrics(
        self,
        episode: Episode,
        helper_actions: List[Action],
    ) -> Dict[str, Any]:
        """Compute task performance metrics (completion, wasted actions, efficiency).

        Args:
            episode: Episode being evaluated
            helper_actions: List of helper actions taken

        Returns:
            Dictionary with task performance metrics
        """
        task_completed = episode.metadata.get("task_completed", False)
        num_steps = len(episode.steps)
        
        # Count wasted actions (simplified heuristic)
        wasted_actions = 0
        goal_id = episode.goal_id
        task = get_task(goal_id)
        
        for step in episode.steps:
            human_action = step.human_action
            # Simplified: MOVE actions when critical objects visible = wasted
            if human_action == Action.MOVE:
                for obj_id in task.critical_objects:
                    if obj_id in step.visible_objects_h:
                        wasted_actions += 1
                        break
        
        # Efficiency: useful actions / total actions
        total_actions = len(episode.steps)
        useful_actions = total_actions - wasted_actions
        efficiency = useful_actions / total_actions if total_actions > 0 else 0.0
        
        return {
            "task_completed": task_completed,
            "num_steps_to_completion": num_steps if task_completed else None,
            "num_wasted_actions": wasted_actions,
            "task_efficiency": efficiency,
            "num_helper_actions": len(helper_actions),
        }

    def _compute_intervention_metrics(
        self,
        episode: Episode,
        helper_actions: List[Action],
        step_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute intervention quality metrics (over/under-correction, precision/recall).

        Args:
            episode: Episode being evaluated
            helper_actions: List of helper actions taken
            step_data: Step-by-step data

        Returns:
            Dictionary with intervention metrics
        """
        # Count interventions (non-WAIT actions)
        interventions = [i for i, action in enumerate(helper_actions) if action != Action.WAIT]
        num_interventions = len(interventions)
        
        # Determine when interventions were needed (when false belief exists)
        intervention_needed = [step["false_belief_exists"] for step in step_data]
        intervention_taken = [action != Action.WAIT for action in helper_actions]
        
        # Over-correction: intervened when not needed
        over_corrections = sum(
            1 for needed, taken in zip(intervention_needed, intervention_taken)
            if not needed and taken
        )
        
        # Under-correction: didn't intervene when needed
        under_corrections = sum(
            1 for needed, taken in zip(intervention_needed, intervention_taken)
            if needed and not taken
        )
        
        # Precision: interventions that were needed / total interventions
        correct_interventions = sum(
            1 for needed, taken in zip(intervention_needed, intervention_taken)
            if needed and taken
        )
        precision = correct_interventions / num_interventions if num_interventions > 0 else 0.0
        
        # Recall: interventions taken when needed / total needed
        total_needed = sum(intervention_needed)
        recall = correct_interventions / total_needed if total_needed > 0 else 0.0
        
        return {
            "num_interventions": num_interventions,
            "over_corrections": over_corrections,
            "under_corrections": under_corrections,
            "intervention_precision": precision,
            "intervention_recall": recall,
        }
