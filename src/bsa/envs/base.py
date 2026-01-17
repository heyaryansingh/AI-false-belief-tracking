"""Base environment interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..common.types import Action, Observation, ObjectLocation


class Environment(ABC):
    """Base environment interface for simulators."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation
        """
        pass

    @abstractmethod
    def step(self, action: Action, agent_id: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute action and advance environment.

        Args:
            action: Action to execute
            agent_id: ID of agent taking action

        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_true_state(self) -> Dict[str, Any]:
        """Get true state of the environment (privileged information).

        Returns:
            Dictionary containing true state information
        """
        pass

    @abstractmethod
    def get_visible_state(self, agent_id: str) -> Observation:
        """Get visible state for an agent (partial observability).

        Args:
            agent_id: ID of agent

        Returns:
            Observation available to agent
        """
        pass

    @abstractmethod
    def get_object_locations(self) -> Dict[str, ObjectLocation]:
        """Get current object locations.

        Returns:
            Dictionary mapping object_id to ObjectLocation
        """
        pass

    def clone(self) -> "Environment":
        """Create a deep copy of the environment (optional).

        Returns:
            Cloned environment
        """
        raise NotImplementedError("clone() not implemented for this environment")
