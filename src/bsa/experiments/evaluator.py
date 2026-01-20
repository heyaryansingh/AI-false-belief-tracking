"""Comprehensive episode evaluator with detailed metrics computation.

# Fix: AUROC computed per-episode, aggregated later with bootstrap CI (Phase 10)
# This ensures proper statistical treatment of classification metrics.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import warnings

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
        false_belief_scores: List[float] = []  # Probability scores for AUROC
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
            
            # Track false-belief detection (with probability scores)
            if hasattr(helper_agent, 'compute_false_belief_confidence'):
                # Use probability-based detection
                detection_confidence = helper_agent.compute_false_belief_confidence(episode_step=step)
                false_belief_predicted = detection_confidence >= 0.5  # Threshold
                false_belief_scores.append(detection_confidence)
            else:
                # Fallback to binary detection
                false_belief_predicted = helper_agent.detect_false_belief(helper_obs, episode_step=step)
                false_belief_scores.append(1.0 if false_belief_predicted else 0.0)
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
            false_belief_predictions, false_belief_ground_truth, step_data, false_belief_scores
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
        scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute false-belief detection metrics (AUROC, detection latency, FPR).

        # Fix: AUROC computed per-episode, aggregated later with bootstrap CI (Phase 10)
        # Scores are validated to be probabilities [0, 1] for proper AUROC computation.

        Args:
            predictions: List of false-belief predictions (one per step)
            ground_truth: List of ground truth false-belief existence (one per step)
            step_data: Step-by-step data
            scores: Optional probability scores for AUROC computation

        Returns:
            Dictionary with detection metrics including temporal metrics
        """
        total_steps = len(predictions)
        
        if not any(ground_truth):
            # No false beliefs in episode - return baseline metrics
            return {
                "false_belief_detection_auroc": None,
                "false_belief_detection_latency": None,
                "false_belief_detection_fpr": None,
                "false_belief_detected": False,
                # Fix: Added temporal metrics (Phase 10)
                "false_belief_onset_timestep": None,
                "time_to_detection": None,
                "false_alarm_rate": 0.0,
                "detection_timesteps": [],
            }
        
        # Fix: Validate scores are probabilities [0, 1] (Phase 10)
        if scores:
            scores_array = np.array(scores)
            if np.any(scores_array < 0) or np.any(scores_array > 1):
                warnings.warn(
                    "Detection scores outside [0, 1] range detected. "
                    "Clamping to valid probability range."
                )
                scores = [max(0.0, min(1.0, s)) for s in scores]
        
        # Compute detection latency (first timestep when false belief detected)
        detection_latency = None
        first_false_belief_timestep = None
        first_detection_timestep = None
        detection_timesteps = []  # Fix: Track all detection timesteps (Phase 10)
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            timestep = step_data[i]["timestep"]
            if gt and first_false_belief_timestep is None:
                first_false_belief_timestep = timestep
            if pred:
                detection_timesteps.append(timestep)
                if first_detection_timestep is None and first_false_belief_timestep is not None:
                    first_detection_timestep = timestep
                    detection_latency = timestep - first_false_belief_timestep
        
        # Fix: Compute time-to-detection (TTD) - mean time from onset to all detections (Phase 10)
        time_to_detection = None
        if first_false_belief_timestep is not None and detection_timesteps:
            valid_detections = [t for t in detection_timesteps if t >= first_false_belief_timestep]
            if valid_detections:
                time_to_detection = np.mean([t - first_false_belief_timestep for t in valid_detections])
        
        # Compute AUROC using probability scores
        # Fix: Per-episode AUROC for later aggregation with bootstrap CI (Phase 10)
        auroc = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(ground_truth)) > 1:  # Need both classes for AUROC
                if scores and len(scores) == len(ground_truth):
                    # Use probability scores for better AUROC discrimination
                    auroc = roc_auc_score(ground_truth, scores)
                else:
                    # Fallback to binary scores (less informative)
                    binary_scores = [1.0 if p else 0.0 for p in predictions]
                    auroc = roc_auc_score(ground_truth, binary_scores)
        except ImportError:
            # sklearn not available
            auroc = None
        except ValueError:
            # Edge case: all predictions same class
            auroc = None
        
        # Compute false positive rate
        false_positives = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
        true_negatives = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        
        # Fix: Compute false alarm rate per episode (Phase 10)
        false_alarm_rate = false_positives / total_steps if total_steps > 0 else 0.0
        
        return {
            "false_belief_detection_auroc": auroc,
            "false_belief_detection_latency": detection_latency,
            "false_belief_detection_fpr": fpr,
            "false_belief_detected": any(predictions),
            # Fix: Added temporal metrics for realistic detection timing (Phase 10)
            "false_belief_onset_timestep": first_false_belief_timestep,
            "time_to_detection": time_to_detection,
            "false_alarm_rate": false_alarm_rate,
            "detection_timesteps": detection_timesteps,
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

        # Fix: Efficiency recalculated per model/episode independently (Phase 10)
        # Each evaluation uses episode-specific data with no shared state.

        Args:
            episode: Episode being evaluated
            helper_actions: List of helper actions taken

        Returns:
            Dictionary with task performance metrics
        """
        task_completed = episode.metadata.get("task_completed", False)
        completion_timestep = episode.metadata.get("completion_timestep", None)
        num_steps = len(episode.steps)
        
        # Count wasted actions with improved heuristics
        # Fix: Ensure wasted_actions is computed independently for each episode (Phase 10)
        wasted_actions = 0.0  # Use float for fractional penalties
        goal_id = episode.goal_id
        task = get_task(goal_id)

        # Track visited rooms to detect backtracking
        visited_rooms = []
        
        # Track wasted action breakdown for debugging
        wasted_move_when_visible = 0
        wasted_failed_pickup = 0
        wasted_backtracking = 0.0

        for i, step in enumerate(episode.steps):
            human_action = step.human_action
            current_room = step.human_obs.current_room if step.human_obs else None

            # Heuristic 1: MOVE when critical object is visible and adjacent
            if human_action == Action.MOVE:
                for obj_id in task.critical_objects:
                    if obj_id in step.visible_objects_h:
                        wasted_actions += 1
                        wasted_move_when_visible += 1
                        break

            # Heuristic 2: PICKUP when no object nearby (failed pickup)
            if human_action == Action.PICKUP:
                # Check if any critical object was actually picked up
                if i + 1 < len(episode.steps):
                    next_step = episode.steps[i + 1]
                    current_objs = set(step.true_object_locations.keys())
                    next_objs = set(next_step.true_object_locations.keys())
                    picked_up = current_objs - next_objs
                    if not any(obj_id in picked_up for obj_id in task.critical_objects):
                        wasted_actions += 1
                        wasted_failed_pickup += 1

            # Heuristic 3: Backtracking (visiting same room multiple times)
            if current_room:
                if len(visited_rooms) >= 2 and visited_rooms[-1] != current_room:
                    if current_room in visited_rooms[:-1]:
                        wasted_actions += 0.5  # Partial penalty for backtracking
                        wasted_backtracking += 0.5
                visited_rooms.append(current_room)
        
        # Efficiency: useful actions / total actions
        # Fix: Computed per-episode, not globally (Phase 10)
        total_actions = len(episode.steps)
        useful_actions = total_actions - wasted_actions
        efficiency = max(0.0, useful_actions / total_actions) if total_actions > 0 else 0.0
        
        # Compute steps to completion
        num_steps_to_completion = None
        if task_completed and completion_timestep is not None:
            num_steps_to_completion = completion_timestep + 1  # +1 because timestep is 0-indexed
        
        # Count helper-specific actions
        helper_wait_count = sum(1 for a in helper_actions if a == Action.WAIT)
        helper_intervention_count = len(helper_actions) - helper_wait_count
        
        return {
            "task_completed": task_completed,
            "num_steps_to_completion": num_steps_to_completion,
            "num_wasted_actions": wasted_actions,
            "task_efficiency": efficiency,
            "num_helper_actions": len(helper_actions),
            # Fix: Added detailed wasted action breakdown (Phase 10)
            "wasted_move_when_visible": wasted_move_when_visible,
            "wasted_failed_pickup": wasted_failed_pickup,
            "wasted_backtracking": wasted_backtracking,
            "helper_wait_count": helper_wait_count,
            "helper_intervention_count": helper_intervention_count,
        }

    def _compute_intervention_metrics(
        self,
        episode: Episode,
        helper_actions: List[Action],
        step_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute intervention quality metrics (over/under-correction, precision/recall).

        # Fix: Temporal precision-recall tracks detection quality over episode (Phase 10)

        Args:
            episode: Episode being evaluated
            helper_actions: List of helper actions taken
            step_data: Step-by-step data

        Returns:
            Dictionary with intervention metrics including temporal tracking
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
        
        # Fix: Compute precision/recall over time (cumulative at each timestep) (Phase 10)
        precision_over_time = []
        recall_over_time = []
        cumulative_correct = 0
        cumulative_interventions = 0
        cumulative_needed = 0
        
        for needed, taken in zip(intervention_needed, intervention_taken):
            if needed:
                cumulative_needed += 1
            if taken:
                cumulative_interventions += 1
                if needed:
                    cumulative_correct += 1
            
            # Compute cumulative precision and recall
            curr_precision = cumulative_correct / cumulative_interventions if cumulative_interventions > 0 else 0.0
            curr_recall = cumulative_correct / cumulative_needed if cumulative_needed > 0 else 0.0
            precision_over_time.append(curr_precision)
            recall_over_time.append(curr_recall)
        
        # F1 score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "num_interventions": num_interventions,
            "over_corrections": over_corrections,
            "under_corrections": under_corrections,
            "intervention_precision": precision,
            "intervention_recall": recall,
            # Fix: Added F1 score and temporal metrics (Phase 10)
            "intervention_f1": f1_score,
            "precision_over_time": precision_over_time,
            "recall_over_time": recall_over_time,
        }
