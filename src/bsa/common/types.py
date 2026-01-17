"""Common type definitions."""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Action types for agents."""

    MOVE = "move"
    OPEN = "open"
    CLOSE = "close"
    PICKUP = "pickup"
    PLACE = "place"
    WAIT = "wait"
    SAY = "say"


@dataclass
class ObjectLocation:
    """Represents an object's location."""

    object_id: str
    container_id: Optional[str]  # None if in room
    room_id: str
    position: Tuple[float, float, float]  # x, y, z coordinates


@dataclass
class Observation:
    """Agent observation."""

    agent_id: str
    visible_objects: List[str]  # Object IDs
    visible_containers: List[str]  # Container IDs
    current_room: str
    position: Tuple[float, float, float]
    timestamp: int


@dataclass
class BeliefState:
    """Belief state over object locations."""

    object_locations: Dict[str, Dict[str, float]]  # object_id -> {location_id: probability}
    goal_distribution: Dict[str, float]  # goal_id -> probability


@dataclass
class EpisodeStep:
    """Single step in an episode."""

    episode_id: str
    timestep: int
    human_action: Action
    helper_obs: Observation
    human_obs: Observation
    visible_objects_h: List[str]
    visible_objects_helper: List[str]
    true_object_locations: Dict[str, ObjectLocation]
    human_belief_object_locations: Dict[str, ObjectLocation]
    goal_id: str
    tau: Optional[int]  # Intervention timestep
    intervention_type: Optional[str]


@dataclass
class Episode:
    """Complete episode."""

    episode_id: str
    goal_id: str
    tau: Optional[int]
    intervention_type: Optional[str]
    steps: List[EpisodeStep]
    metadata: Dict[str, Union[str, int, float]]


@dataclass
class Task:
    """Task definition."""

    task_id: str
    name: str
    description: str
    critical_objects: List[str]  # Objects required for task
    goal_locations: Dict[str, str]  # object_id -> target_location_id


class Goal(Enum):
    """Task goals."""

    PREPARE_MEAL = "prepare_meal"
    SET_TABLE = "set_table"
    PACK_BAG = "pack_bag"
    FIND_KEYS = "find_keys"
