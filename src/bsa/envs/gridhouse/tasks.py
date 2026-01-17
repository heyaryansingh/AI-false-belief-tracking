"""Task definitions for GridHouse."""

from typing import Dict, List
from ...common.types import Task, Goal


# Task definitions
TASKS: Dict[str, Task] = {
    "prepare_meal": Task(
        task_id="prepare_meal",
        name="Prepare Meal",
        description="Prepare a meal using kitchen tools",
        critical_objects=["knife", "plate"],
        goal_locations={"knife": "cabinet_kitchen", "plate": "cabinet_kitchen"},
    ),
    "set_table": Task(
        task_id="set_table",
        name="Set Table",
        description="Set the table for a meal",
        critical_objects=["plate", "fork"],
        goal_locations={"plate": "table_living", "fork": "drawer_kitchen"},
    ),
    "pack_bag": Task(
        task_id="pack_bag",
        name="Pack Bag",
        description="Pack items into a bag",
        critical_objects=["book", "keys"],
        goal_locations={"book": "desk_bedroom", "keys": "desk_bedroom"},
    ),
    "find_keys": Task(
        task_id="find_keys",
        name="Find Keys",
        description="Find and retrieve keys",
        critical_objects=["keys"],
        goal_locations={"keys": "desk_bedroom"},
    ),
}


def get_task(task_id: str) -> Task:
    """Get task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} not found")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """List all available tasks."""
    return list(TASKS.keys())
