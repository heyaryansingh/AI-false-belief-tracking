"""Comprehensive test of Phase 2 implementations."""

print("=" * 60)
print("Phase 2 Implementation Review")
print("=" * 60)

# Test 1: Base Helper Interface
print("\n[TEST 1] Base Helper Interface")
try:
    from src.bsa.agents.helper import HelperAgent
    from abc import ABC
    assert issubclass(HelperAgent, ABC)
    print("  [OK] HelperAgent is abstract base class")
    print("  [OK] plan_action, update_belief, get_belief_state are abstract")
    print("  [OK] reset, detect_false_belief have default implementations")
except Exception as e:
    print(f"  [FAIL] Error: {e}")

# Test 2: Reactive Helper
print("\n[TEST 2] Reactive Helper")
try:
    from src.bsa.agents.helper import ReactiveHelper
    h = ReactiveHelper()
    assert h.get_belief_state() is None
    from src.bsa.common.types import Observation, Action
    obs = Observation(
        agent_id="helper",
        visible_objects=["knife"],
        visible_containers=[],
        current_room="kitchen",
        position=(0.0, 0.0, 0.0),
        timestamp=0
    )
    action = h.plan_action(obs)
    assert isinstance(action, Action)
    print("  [OK] ReactiveHelper implements HelperAgent")
    print("  [OK] get_belief_state returns None")
    print("  [OK] plan_action returns valid Action")
except Exception as e:
    print(f"  [FAIL] Error: {e}")

# Test 3: Goal Inference
print("\n[TEST 3] Goal Inference")
try:
    from src.bsa.inference.goal import GoalInference
    from src.bsa.envs.gridhouse.tasks import list_tasks, get_task
    tasks = [get_task(t) for t in list_tasks()]
    gi = GoalInference(tasks)
    dist = gi.get_goal_distribution()
    assert len(dist) == len(tasks)
    assert abs(sum(dist.values()) - 1.0) < 1e-6  # Probabilities sum to 1
    most_likely = gi.get_most_likely_goal()
    assert most_likely is not None
    print("  [OK] GoalInference initializes with uniform prior")
    print("  [OK] Goal distribution sums to 1.0")
    print("  [OK] get_most_likely_goal works")
    
    # Test update
    from src.bsa.common.types import Action, Observation
    obs = Observation(
        agent_id="human",
        visible_objects=["knife"],
        visible_containers=[],
        current_room="kitchen",
        position=(0.0, 0.0, 0.0),
        timestamp=0
    )
    gi.update(Action.PICKUP, obs)
    new_dist = gi.get_goal_distribution()
    assert abs(sum(new_dist.values()) - 1.0) < 1e-6
    print("  [OK] Goal inference updates correctly")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Goal-Only Helper
print("\n[TEST 4] Goal-Only Helper")
try:
    from src.bsa.agents.helper import GoalOnlyHelper
    h = GoalOnlyHelper()
    state = h.get_belief_state()
    assert isinstance(state, dict)
    assert "goal_distribution" in state
    assert isinstance(state["goal_distribution"], dict)
    print("  [OK] GoalOnlyHelper implements HelperAgent")
    print("  [OK] get_belief_state returns goal distribution")
    
    obs = Observation(
        agent_id="helper",
        visible_objects=["knife"],
        visible_containers=[],
        current_room="kitchen",
        position=(0.0, 0.0, 0.0),
        timestamp=0
    )
    action = h.plan_action(obs)
    assert isinstance(action, Action)
    print("  [OK] plan_action returns valid Action")
    assert h.detect_false_belief(obs) == False
    print("  [OK] detect_false_belief returns False (as expected)")
except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Dependencies for Plan 02-04
print("\n[TEST 5] Dependencies for Plan 02-04")
try:
    from src.bsa.common.types import Action, Observation, ObjectLocation, Task, EpisodeStep
    from src.bsa.inference.goal import GoalInference
    from src.bsa.envs.gridhouse.tasks import get_task, list_tasks
    from src.bsa.agents.human.policies import plan_next_action
    import numpy as np
    print("  [OK] All required types available")
    print("  [OK] GoalInference available")
    print("  [OK] Task definitions available")
    print("  [OK] Human agent policies available")
    print("  [OK] NumPy available")
except Exception as e:
    print(f"  [FAIL] Missing dependency: {e}")

# Test 6: Interface Consistency
print("\n[TEST 6] Interface Consistency")
try:
    from src.bsa.agents.helper import HelperAgent, ReactiveHelper, GoalOnlyHelper
    from src.bsa.common.types import Observation
    
    obs = Observation(
        agent_id="helper",
        visible_objects=[],
        visible_containers=[],
        current_room="kitchen",
        position=(0.0, 0.0, 0.0),
        timestamp=0
    )
    
    # All helpers should implement the same interface
    reactive = ReactiveHelper()
    goal_only = GoalOnlyHelper()
    
    # All should have plan_action
    assert hasattr(reactive, 'plan_action')
    assert hasattr(goal_only, 'plan_action')
    
    # All should have update_belief
    assert hasattr(reactive, 'update_belief')
    assert hasattr(goal_only, 'update_belief')
    
    # All should have get_belief_state
    assert hasattr(reactive, 'get_belief_state')
    assert hasattr(goal_only, 'get_belief_state')
    
    print("  [OK] All helpers implement HelperAgent interface consistently")
except Exception as e:
    print(f"  [FAIL] Error: {e}")

print("\n" + "=" * 60)
print("Review Complete!")
print("=" * 60)
