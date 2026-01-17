"""Planning policies for human agent."""

from typing import Dict, List, Optional, Tuple
import math

from ...common.types import Action, ObjectLocation, Task


def plan_next_action(
    agent_pos: Tuple[int, int],
    agent_room: str,
    task: Task,
    belief_state: Dict[str, ObjectLocation],
    visible_objects: List[str],
    visible_containers: List[str],
) -> Action:
    """Plan next action based on task goal and belief state.

    Uses simple heuristic: move toward closest critical object needed for task.

    Args:
        agent_pos: Current agent position (x, y)
        agent_room: Current room ID
        task: Task to accomplish
        belief_state: Beliefs about object locations
        visible_objects: List of object IDs currently visible
        visible_containers: List of container IDs currently visible

    Returns:
        Next action to take
    """
    # If we can see a critical object, try to pick it up
    for obj_id in task.critical_objects:
        if obj_id in visible_objects:
            # Object is visible - try to pick it up
            # (In practice, we'd check distance, but for now assume we can pickup)
            return Action.PICKUP

    # No visible critical objects - plan path to believed location
    # Find closest critical object based on belief state
    closest_obj = None
    closest_distance = float("inf")
    target_room = None
    target_container = None

    for obj_id in task.critical_objects:
        if obj_id not in belief_state:
            continue  # No belief about this object yet

        obj_belief = belief_state[obj_id]
        obj_room = obj_belief.room_id
        obj_container = obj_belief.container_id

        # Calculate distance (simple: different room = far, same room = close)
        if obj_room == agent_room:
            # Same room - check if we need to open container
            if obj_container:
                # Object is in a container
                if obj_container in visible_containers:
                    # Container is visible - check if it's open
                    # For now, assume we need to open it if object is inside
                    distance = 1  # Close enough to interact
                else:
                    distance = 2  # Need to move to container
            else:
                # Object is in room (not in container)
                distance = 1  # Close
        else:
            # Different room - far
            distance = 10

        if distance < closest_distance:
            closest_distance = distance
            closest_obj = obj_id
            target_room = obj_room
            target_container = obj_container

    # Plan action based on closest object
    if closest_obj is None:
        # No beliefs about critical objects - explore
        return Action.MOVE

    if target_room != agent_room:
        # Need to move to different room
        return Action.MOVE

    if target_container:
        # Object is in a container
        if target_container in visible_containers:
            # Container is visible - try to open it
            return Action.OPEN
        else:
            # Need to move toward container
            return Action.MOVE

    # Same room, object should be visible if not in container
    # Try to pick it up
    return Action.PICKUP
