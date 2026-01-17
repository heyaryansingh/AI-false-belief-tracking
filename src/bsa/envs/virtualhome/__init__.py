"""VirtualHome adapter."""

try:
    from .env import VirtualHomeEnvironment
    from .tasks import get_task, list_tasks
    from .episode_generator import VirtualHomeEpisodeGenerator

    __all__ = [
        "VirtualHomeEnvironment",
        "get_task",
        "list_tasks",
        "VirtualHomeEpisodeGenerator",
    ]
except ImportError:
    # VirtualHome not installed - export empty
    __all__ = []
