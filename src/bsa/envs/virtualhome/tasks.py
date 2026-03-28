"""Tasks for VirtualHome."""

from typing import List, Dict
from ...common.types import Task

def get_vh_task_list() -> List[Task]:
    return [
        Task("find_keys", "Find Keys", "Find the keys and pick them up.", ["keys"], {}),
        Task("find_phone", "Find Phone", "Find the phone.", ["phone"], {}),
        Task("find_book", "Find Book", "Find and read a book.", ["book"], {})
    ]
