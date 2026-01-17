"""VirtualHome adapter."""

try:
    from .env import VirtualHomeEnvironment
    from .tasks import get_task, list_tasks
    from .episode_generator import VirtualHomeEpisodeGenerator
    from .recorder import VirtualHomeEpisodeRecorder
    from .observability import (
        get_scene_state,
        get_agent_view,
        get_object_trajectory,
        analyze_observability,
        visualize_episode,
    )

    __all__ = [
        "VirtualHomeEnvironment",
        "get_task",
        "list_tasks",
        "VirtualHomeEpisodeGenerator",
        "VirtualHomeEpisodeRecorder",
        "get_scene_state",
        "get_agent_view",
        "get_object_trajectory",
        "analyze_observability",
        "visualize_episode",
    ]
except ImportError:
    # VirtualHome not installed - export empty
    __all__ = []
