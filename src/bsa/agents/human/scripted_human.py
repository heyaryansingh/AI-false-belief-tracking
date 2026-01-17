"""Scripted human agent with belief tracking."""

from typing import Dict, Optional
from ...common.types import Action, Observation, ObjectLocation, Task


class ScriptedHumanAgent:
    """Human agent that plans actions based on task goals and belief state.

    The agent maintains beliefs about object locations and updates them only
    when observing evidence. This enables false-belief scenarios when objects
    move without the agent observing.
    """

    def __init__(self, goal: Task, initial_belief_state: Optional[Dict[str, ObjectLocation]] = None):
        """Initialize human agent.

        Args:
            goal: Task goal the agent is trying to accomplish
            initial_belief_state: Initial beliefs about object locations.
                If None, beliefs start empty (will be initialized from observation)
        """
        self.goal = goal
        self.belief_state: Dict[str, ObjectLocation] = initial_belief_state.copy() if initial_belief_state else {}

    def plan_action(
        self,
        observation: Observation,
        task: Optional[Task] = None,
        belief_state: Optional[Dict[str, ObjectLocation]] = None,
    ) -> Action:
        """Plan next action based on observation, task, and belief state.

        Args:
            observation: Current observation from environment
            task: Task to accomplish (if None, uses self.goal)
            belief_state: Belief state about object locations (if None, uses self.belief_state)

        Returns:
            Next action to take
        """
        from .policies import plan_next_action

        task = task or self.goal
        belief_state = belief_state or self.belief_state

        agent_pos = (int(observation.position[0]), int(observation.position[1]))
        agent_room = observation.current_room

        return plan_next_action(
            agent_pos=agent_pos,
            agent_room=agent_room,
            task=task,
            belief_state=belief_state,
            visible_objects=observation.visible_objects,
            visible_containers=observation.visible_containers,
        )

    def update_belief(
        self,
        observation: Observation,
        true_object_locations: Dict[str, ObjectLocation],
        belief_state: Optional[Dict[str, ObjectLocation]] = None,
    ) -> None:
        """Update belief state based on observation.

        Only updates beliefs for objects that are visible in the observation.
        This enables false beliefs when objects move without being observed.

        Args:
            observation: Current observation from environment
            true_object_locations: True object locations from environment
            belief_state: Belief state to update (if None, updates self.belief_state)
        """
        belief_state = belief_state or self.belief_state

        # Update beliefs only for visible objects
        # Objects in containers are only visible if container is open
        for obj_id in observation.visible_objects:
            if obj_id in true_object_locations:
                # Object is visible - update belief to match true location
                belief_state[obj_id] = true_object_locations[obj_id]
