"""Planning policies for human agent.

FIXED VERSION: Goal-directed behavior that actually completes tasks.
- Moves toward objects based on belief state
- Only attempts pickup when adjacent to object
- Opens containers to access objects inside
"""

from typing import Dict, List, Optional, Tuple
import math

from ...common.types import Action, ObjectLocation, Task


# Room center coordinates for navigation
ROOM_CENTERS = {
    "kitchen": (5, 5),
    "living_room": (15, 5),
    "bedroom": (5, 15),
    "bathroom": (15, 15),
}

# Container positions
CONTAINER_POSITIONS = {
    "cabinet_kitchen": (5, 5),
    "drawer_kitchen": (7, 5),
    "table_living": (15, 5),
    "desk_bedroom": (5, 15),
}


def plan_next_action(
    agent_pos: Tuple[int, int],
    agent_room: str,
    task: Task,
    belief_state: Dict[str, ObjectLocation],
    visible_objects: List[str],
    visible_containers: List[str],
) -> Action:
    """Plan next action based on task goal and belief state.

    FIXED: Proper goal-directed planning:
    1. Find target critical object (prioritize visible, then nearest)
    2. If adjacent to object -> PICKUP
    3. If object in closed container we can see -> OPEN
    4. Otherwise -> MOVE toward object

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
    # Find target object to pursue
    target_obj_id = None
    target_obj_loc = None

    # Priority 1: Visible critical object
    for obj_id in task.critical_objects:
        if obj_id in visible_objects:
            if obj_id in belief_state:
                target_obj_id = obj_id
                target_obj_loc = belief_state[obj_id]
                break

    # Priority 2: Closest critical object based on belief
    if target_obj_id is None:
        closest_distance = float("inf")
        for obj_id in task.critical_objects:
            if obj_id not in belief_state:
                continue

            obj_loc = belief_state[obj_id]
            distance = _estimate_distance(agent_pos, agent_room, obj_loc)

            if distance < closest_distance:
                closest_distance = distance
                target_obj_id = obj_id
                target_obj_loc = obj_loc

    # No target - explore
    if target_obj_id is None or target_obj_loc is None:
        return Action.MOVE

    # Check if we're adjacent to the object (can pickup)
    if target_obj_id in visible_objects:
        obj_distance = _point_distance(
            agent_pos,
            (int(target_obj_loc.position[0]), int(target_obj_loc.position[1]))
        )
        if obj_distance <= 1.5:  # Adjacent - pickup
            return Action.PICKUP

    # Check if object is in a container we need to open
    if target_obj_loc.container_id:
        container_id = target_obj_loc.container_id
        if container_id in visible_containers:
            # Container is visible - check if we need to open it
            # If object not visible but we believe it's in this container, open it
            if target_obj_id not in visible_objects:
                return Action.OPEN
        else:
            # Container not visible - move toward it
            container_pos = CONTAINER_POSITIONS.get(container_id)
            if container_pos:
                return _move_toward(agent_pos, container_pos)

    # Object should be accessible - move toward it
    target_pos = (int(target_obj_loc.position[0]), int(target_obj_loc.position[1]))

    # If in wrong room, move toward that room
    if target_obj_loc.room_id != agent_room:
        target_room_center = ROOM_CENTERS.get(target_obj_loc.room_id, target_pos)
        return _move_toward(agent_pos, target_room_center)

    # In same room - move toward object
    return _move_toward(agent_pos, target_pos)


def _estimate_distance(
    agent_pos: Tuple[int, int],
    agent_room: str,
    obj_loc: ObjectLocation,
) -> float:
    """Estimate distance to object, considering room boundaries."""
    obj_pos = (int(obj_loc.position[0]), int(obj_loc.position[1]))

    # Same room - direct distance
    if obj_loc.room_id == agent_room:
        return _point_distance(agent_pos, obj_pos)

    # Different room - add room transition penalty
    return _point_distance(agent_pos, obj_pos) + 10.0


def _point_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _move_toward(current: Tuple[int, int], target: Tuple[int, int]) -> Action:
    """Return MOVE action (actual direction handled by environment).

    In a more sophisticated implementation, this would return a
    directional move. For now, we just return MOVE and let the
    environment handle movement.
    """
    # TODO: Add directional movement when environment supports it
    return Action.MOVE
