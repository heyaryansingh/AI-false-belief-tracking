"""Task definitions for VirtualHome."""

from typing import Dict, List
from ...common.types import Task


# Task definitions for VirtualHome scenes
# Object names match VirtualHome scene object naming conventions
TASKS: Dict[str, Task] = {
    "prepare_meal": Task(
        task_id="prepare_meal",
        name="Prepare Meal",
        description="Prepare a meal using kitchen tools",
        critical_objects=["knife", "plate", "apple"],  # VirtualHome object names
        goal_locations={"knife": "kitchen", "plate": "kitchen", "apple": "kitchen"},
    ),
    "set_table": Task(
        task_id="set_table",
        name="Set Table",
        description="Set the table for a meal",
        critical_objects=["plate", "fork", "knife"],
        goal_locations={"plate": "dining_room", "fork": "dining_room", "knife": "dining_room"},
    ),
    "pack_bag": Task(
        task_id="pack_bag",
        name="Pack Bag",
        description="Pack items into a bag",
        critical_objects=["book", "keys"],
        goal_locations={"book": "bedroom", "keys": "bedroom"},
    ),
    "find_keys": Task(
        task_id="find_keys",
        name="Find Keys",
        description="Find and retrieve keys",
        critical_objects=["keys"],
        goal_locations={"keys": "bedroom"},
    ),
}


def get_task(task_id: str) -> Task:
    """Get task by ID.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task object
        
    Raises:
        ValueError: If task_id not found
    """
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} not found. Available tasks: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """List all available task IDs.
    
    Returns:
        List of task IDs
    """
    return list(TASKS.keys())
