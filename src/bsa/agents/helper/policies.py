"""Intervention policies for belief-sensitive helpers."""

from typing import Any, Dict

from ...common.types import Action, ObjectLocation, Observation, Task
from ...inference.belief import BeliefInference


class InterventionPolicy:
    """Policy for deciding when and how to intervene to assist human.

    Determines:
    - When intervention is needed (false belief detected, human wasting actions)
    - What type of intervention to take (fetch, communicate, open, wait)
    - Specific action to execute
    """

    def should_intervene(
        self,
        belief_state: Dict[str, Any],
        true_locations: Dict[str, ObjectLocation],
        observation: Observation,
        task: Task,
    ) -> bool:
        """Decide if intervention is needed.

        Returns True if:
        - False belief detected AND human is likely to waste actions
        - Human needs assistance with goal (no false belief but can help)

        Args:
            belief_state: Current belief state from BeliefInference
            true_locations: True object locations
            observation: Current observation
            task: Inferred task/goal

        Returns:
            True if intervention is needed, False otherwise
        """
        # Check for false belief
        # Extract believed locations from belief_state
        object_location_beliefs = belief_state.get("object_location_beliefs", {})

        # Check if any critical object has false belief
        for obj_id in task.critical_objects:
            if obj_id in true_locations:
                true_loc = true_locations[obj_id]
                # Get most likely believed location
                if obj_id in object_location_beliefs:
                    believed_locs = object_location_beliefs[obj_id]
                    if believed_locs:
                        # Get most likely location
                        most_likely_loc = max(believed_locs.items(), key=lambda x: x[1])[0]
                        if most_likely_loc != true_loc.room_id:
                            return True  # False belief detected - intervene

        # Also intervene if we can help with goal (fetch objects, open containers)
        # Check if critical objects are visible or in containers
        for obj_id in task.critical_objects:
            if obj_id in observation.visible_objects:
                return True  # Can fetch visible object
            # Check if object is in visible container
            if obj_id in true_locations:
                obj_loc = true_locations[obj_id]
                if obj_loc.container_id and obj_loc.container_id in observation.visible_containers:
                    return True  # Can open container

        return False  # No intervention needed

    def choose_intervention(
        self,
        belief_state: Dict[str, Any],
        true_locations: Dict[str, ObjectLocation],
        observation: Observation,
        task: Task,
    ) -> Action:
        """Choose intervention action.

        Policy:
        - If false belief: fetch object from true location or communicate
        - If goal unclear: wait or explore
        - If no false belief: assist with goal (fetch objects, open containers)

        Args:
            belief_state: Current belief state
            true_locations: True object locations
            observation: Current observation
            task: Inferred task/goal

        Returns:
            Action to take
        """
        # Check for false belief first
        object_location_beliefs = belief_state.get("object_location_beliefs", {})

        for obj_id in task.critical_objects:
            if obj_id in true_locations:
                true_loc = true_locations[obj_id]
                # Check if false belief
                if obj_id in object_location_beliefs:
                    believed_locs = object_location_beliefs[obj_id]
                    if believed_locs:
                        most_likely_loc = max(believed_locs.items(), key=lambda x: x[1])[0]
                        if most_likely_loc != true_loc.room_id:
                            # False belief detected - fetch object from true location
                            if obj_id in observation.visible_objects:
                                return Action.PICKUP  # Fetch object
                            else:
                                # Object not visible - move towards it or communicate
                                return Action.SAY  # Communicate correction

        # No false belief - assist with goal
        # If critical objects are visible, pick them up
        for obj_id in task.critical_objects:
            if obj_id in observation.visible_objects:
                return Action.PICKUP

        # If containers with critical objects are visible, open them
        for obj_id in task.critical_objects:
            if obj_id in true_locations:
                obj_loc = true_locations[obj_id]
                if obj_loc.container_id and obj_loc.container_id in observation.visible_containers:
                    return Action.OPEN

        # Otherwise, move towards goal
        return Action.MOVE

    def get_intervention_type(
        self,
        belief_state: Dict[str, Any],
        true_locations: Dict[str, ObjectLocation],
    ) -> str:
        """Get intervention type.

        Args:
            belief_state: Current belief state
            true_locations: True object locations

        Returns:
            Intervention type: "fetch", "communicate", "open", "wait", or "assist"
        """
        # Simplified: determine type based on action that would be chosen
        # This is a helper method for logging/debugging
        object_location_beliefs = belief_state.get("object_location_beliefs", {})

        # Check for false belief
        for obj_id, true_loc in true_locations.items():
            if obj_id in object_location_beliefs:
                believed_locs = object_location_beliefs[obj_id]
                if believed_locs:
                    most_likely_loc = max(believed_locs.items(), key=lambda x: x[1])[0]
                    if most_likely_loc != true_loc.room_id:
                        return "communicate"  # False belief - communicate

        return "assist"  # No false belief - assist with goal
