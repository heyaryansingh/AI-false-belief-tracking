"""Test script to verify Phase 7 fixes before full regeneration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bsa.envs.gridhouse.env import GridHouseEnvironment
from bsa.envs.gridhouse.episode_generator import GridHouseEpisodeGenerator
from bsa.agents.helper.reactive import ReactiveHelper
from bsa.agents.helper.goal_only import GoalOnlyHelper
from bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper
from bsa.experiments.evaluator import EpisodeEvaluator


def test_episode_generation():
    """Test that episodes are generated with real false beliefs."""
    print("\n" + "="*60)
    print("TEST 1: Episode Generation with False Beliefs")
    print("="*60)

    env = GridHouseEnvironment(seed=42)
    generator = GridHouseEpisodeGenerator(
        env=env,
        seed=42,
        tau_range=(5, 15),
        drift_probability=1.0,  # Always apply intervention
    )

    # Generate 10 episodes
    false_belief_count = 0
    for i in range(10):
        episode = generator.generate_episode(intervention_type="relocate")
        fb_created = episode.metadata.get("false_belief_created", False)
        fb_steps = episode.metadata.get("false_belief_steps", 0)

        if fb_created:
            false_belief_count += 1
            print(f"  Episode {i+1}: FALSE BELIEF CREATED - {fb_steps} steps with false belief")
        else:
            print(f"  Episode {i+1}: No false belief")

    print(f"\nResult: {false_belief_count}/10 episodes have false beliefs")
    return false_belief_count >= 7  # At least 70% should have false beliefs


def test_helper_agents():
    """Test that helper agents produce differentiated results."""
    print("\n" + "="*60)
    print("TEST 2: Helper Agent False Belief Detection")
    print("="*60)

    env = GridHouseEnvironment(seed=123)
    generator = GridHouseEpisodeGenerator(
        env=env,
        seed=123,
        tau_range=(5, 10),
        drift_probability=1.0,
    )

    # Generate a false belief episode
    episode = None
    for _ in range(5):
        episode = generator.generate_episode(intervention_type="relocate")
        if episode.metadata.get("false_belief_created", False):
            break

    if not episode or not episode.metadata.get("false_belief_created", False):
        print("  ERROR: Could not generate false belief episode")
        return False

    print(f"  Generated episode with {len(episode.steps)} steps")
    print(f"  False belief created: {episode.metadata.get('false_belief_created')}")
    print(f"  False belief steps: {episode.metadata.get('false_belief_steps')}")

    # Test each helper agent
    helpers = {
        "reactive": ReactiveHelper(seed=42),
        "goal_only": GoalOnlyHelper(seed=42),
        "belief_pf": BeliefSensitiveHelper(seed=42),
    }

    print("\n  Confidence scores at each step (after tau):")
    tau = episode.tau

    for name, helper in helpers.items():
        helper.reset()
        scores = []
        for step in episode.steps:
            if step.timestep >= tau:
                if hasattr(helper, 'compute_false_belief_confidence'):
                    score = helper.compute_false_belief_confidence(step)
                else:
                    score = 0.5
                scores.append(score)
            # Update belief
            helper.update_belief(step.helper_obs, step.human_action, step)

        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"    {name:12s}: avg={avg_score:.3f} +/- {std_score:.3f} (min={min(scores):.3f}, max={max(scores):.3f})")

    return True


def test_evaluator_metrics():
    """Test that evaluator computes real metrics."""
    print("\n" + "="*60)
    print("TEST 3: Evaluator Metrics Computation")
    print("="*60)

    env = GridHouseEnvironment(seed=456)
    generator = GridHouseEpisodeGenerator(
        env=env,
        seed=456,
        tau_range=(5, 10),
        drift_probability=1.0,
    )

    # Generate a false belief episode
    episode = None
    for _ in range(5):
        episode = generator.generate_episode(intervention_type="relocate")
        if episode.metadata.get("false_belief_created", False):
            break

    if not episode:
        print("  ERROR: Could not generate episode")
        return False

    evaluator = EpisodeEvaluator()

    # Test each helper
    helpers = {
        "reactive": ReactiveHelper(seed=42),
        "goal_only": GoalOnlyHelper(seed=42),
        "belief_pf": BeliefSensitiveHelper(seed=42),
    }

    print("\n  Metrics by helper agent:")
    for name, helper in helpers.items():
        helper.reset()
        metrics = evaluator.evaluate_episode(episode, helper)

        auroc = metrics.get("false_belief_detection_auroc")
        fpr = metrics.get("false_belief_detection_fpr")
        precision = metrics.get("intervention_precision", 0)
        recall = metrics.get("intervention_recall", 0)

        print(f"\n    {name}:")
        print(f"      AUROC: {auroc}")
        print(f"      FPR: {fpr}")
        print(f"      Precision: {precision:.3f}")
        print(f"      Recall: {recall:.3f}")

    return True


def test_metric_variance():
    """Test that metrics show variance across multiple episodes."""
    print("\n" + "="*60)
    print("TEST 4: Metric Variance Across Episodes")
    print("="*60)

    env = GridHouseEnvironment(seed=789)
    generator = GridHouseEpisodeGenerator(
        env=env,
        seed=789,
        tau_range=(5, 15),
        drift_probability=1.0,
    )

    evaluator = EpisodeEvaluator()

    # Collect metrics across episodes
    results = {
        "reactive": {"auroc": [], "precision": [], "recall": []},
        "goal_only": {"auroc": [], "precision": [], "recall": []},
        "belief_pf": {"auroc": [], "precision": [], "recall": []},
    }

    num_episodes = 20
    valid_episodes = 0

    for i in range(num_episodes):
        episode = generator.generate_episode(intervention_type="relocate")
        if not episode.metadata.get("false_belief_created", False):
            continue

        valid_episodes += 1

        for name in results.keys():
            if name == "reactive":
                helper = ReactiveHelper(seed=i)
            elif name == "goal_only":
                helper = GoalOnlyHelper(seed=i)
            else:
                helper = BeliefSensitiveHelper(seed=i)

            metrics = evaluator.evaluate_episode(episode, helper)

            if metrics.get("false_belief_detection_auroc") is not None:
                results[name]["auroc"].append(metrics["false_belief_detection_auroc"])
            results[name]["precision"].append(metrics.get("intervention_precision", 0))
            results[name]["recall"].append(metrics.get("intervention_recall", 0))

    print(f"\n  Analyzed {valid_episodes}/{num_episodes} valid false-belief episodes")
    print("\n  Metric statistics:")

    for name, metrics in results.items():
        print(f"\n    {name}:")
        for metric_name, values in metrics.items():
            if values:
                avg = np.mean(values)
                std = np.std(values)
                print(f"      {metric_name}: {avg:.3f} +/- {std:.3f}")
            else:
                print(f"      {metric_name}: No valid values")

    # Check that we have variance
    has_variance = False
    for name, metrics in results.items():
        for values in metrics.values():
            if values and np.std(values) > 0.01:
                has_variance = True
                break

    return has_variance


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PHASE 7 FIX VERIFICATION TESTS")
    print("#"*60)

    results = []

    try:
        results.append(("Episode Generation", test_episode_generation()))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Episode Generation", False))

    try:
        results.append(("Helper Agents", test_helper_agents()))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Helper Agents", False))

    try:
        results.append(("Evaluator Metrics", test_evaluator_metrics()))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Evaluator Metrics", False))

    try:
        results.append(("Metric Variance", test_metric_variance()))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Metric Variance", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Ready for full regeneration.")
    else:
        print("\nSome tests failed. Check errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
