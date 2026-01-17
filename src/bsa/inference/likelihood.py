"""Likelihood models for action-to-belief inference."""

from abc import ABC, abstractmethod
from typing import Dict

from ..common.types import Action, ObjectLocation, Observation, Task


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models.

    Likelihood models compute P(action | goal, believed_locations, observation),
    which is used in particle filtering to weight particles based on how well
    they explain observed human actions.
    """

    @abstractmethod
    def compute(
        self,
        action: Action,
        goal: Task,
        believed_locations: Dict[str, ObjectLocation],
        observation: Observation,
    ) -> float:
        """Compute P(action | goal, believed_locations, observation).

        Args:
            action: Human action to evaluate
            goal: Task goal hypothesis
            believed_locations: Human's believed object locations (from particle)
            observation: Current observation

        Returns:
            Likelihood P(action | goal, believed_locations, observation) in [0, 1]
        """
        pass


class RuleBasedLikelihoodModel(LikelihoodModel):
    """Rule-based likelihood model for P(action | goal, believed_locations).

    Uses heuristics to determine how likely an action is given:
    - The goal the human is trying to accomplish
    - Where the human believes objects are located
    - What the human can currently observe

    Actions that make progress towards the goal given the believed locations
    have higher likelihood.
    """

    def compute(
        self,
        action: Action,
        goal: Task,
        believed_locations: Dict[str, ObjectLocation],
        observation: Observation,
    ) -> float:
        """Compute likelihood using rule-based heuristics.

        Args:
            action: Human action
            goal: Task goal hypothesis
            believed_locations: Human's believed object locations
            observation: Current observation

        Returns:
            Likelihood in [0, 1]
        """
        agent_room = observation.current_room

        # High likelihood: Actions consistent with goal and believed locations
        if action == Action.PICKUP:
            # Picking up objects - check if critical objects are visible
            for obj_id in goal.critical_objects:
                if obj_id in observation.visible_objects:
                    # High likelihood if picking up critical object
                    return 0.9

        if action == Action.MOVE:
            # Moving - check if moving towards believed locations of critical objects
            for obj_id in goal.critical_objects:
                if obj_id in believed_locations:
                    believed_loc = believed_locations[obj_id]
                    # If moving and object is believed to be in different room,
                    # likely moving towards it
                    if believed_loc.room_id != agent_room:
                        return 0.8  # High likelihood
                    else:
                        return 0.6  # Medium likelihood (moving within room)
            # Moving but no beliefs about critical objects
            return 0.5  # Medium likelihood

        if action == Action.OPEN:
            # Opening containers - check if critical objects might be inside
            for obj_id in goal.critical_objects:
                if obj_id in believed_locations:
                    believed_loc = believed_locations[obj_id]
                    # If object is believed to be in a container
                    if believed_loc.container_id:
                        # Check if container is visible
                        if believed_loc.container_id in observation.visible_containers:
                            return 0.9  # High likelihood
                        else:
                            return 0.7  # Medium-high likelihood
            # Opening container but no beliefs about objects inside
            return 0.6  # Medium likelihood

        if action == Action.PLACE:
            # Placing objects - check if at goal location
            # Simplified: high likelihood if goal involves placing
            return 0.8

        # Low likelihood for actions unrelated to goal
        if action == Action.WAIT:
            return 0.3

        if action == Action.CLOSE:
            return 0.4

        if action == Action.SAY:
            return 0.2  # Communication actions less informative

        # Default: medium likelihood
        return 0.5
