"""VirtualHome environment adapter.

Implements the Environment interface for VirtualHome simulator.
Uses VirtualHome's Evolving Graph mode (pure Python, no Unity required).
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import Environment
from ...common.types import Action, Observation, ObjectLocation
from ...common.seeding import get_rng


class VirtualHomeEnvironment(Environment):
    """VirtualHome environment adapter implementing Environment interface.

    Uses VirtualHome's Evolving Graph simulator for pure-Python execution.
    Provides partial observability based on agent's current room and visibility.
    """

    def __init__(
        self,
        scene_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize VirtualHome environment.

        Args:
            scene_name: Name of VirtualHome scene to load (if None, uses default)
            seed: Random seed for reproducibility
        """
        self.scene_name = scene_name or "FloorPlan1"
        self.rng = get_rng(seed)
        self.timestep = 0
        
        # Try to import VirtualHome
        try:
            from virtualhome.simulation import evolvinggraph
            self._virtualhome_available = True
            self._evolvinggraph = evolvinggraph
        except ImportError:
            self._virtualhome_available = False
            raise ImportError(
                "VirtualHome is not installed. "
                "Install with: pip install virtualhome>=2.3.0\n"
                "Or use GridHouseEnvironment as a fallback."
            )
        
        # VirtualHome scene instance (initialized in reset)
        self._scene = None
        self._graph = None
        
        # Agent tracking: agent_id -> (room_id, position, held_objects)
        self._agents: Dict[str, Dict[str, Any]] = {}
        
        # Object locations cache: object_id -> ObjectLocation
        self._object_locations_cache: Dict[str, ObjectLocation] = {}

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility (if None, uses instance seed)

        Returns:
            Initial observation for default agent
        """
        if seed is not None:
            self.rng = get_rng(seed)
        
        self.timestep = 0
        
        # Initialize VirtualHome scene
        if not self._virtualhome_available:
            raise RuntimeError("VirtualHome not available - cannot reset environment")
        
        # Create or reset VirtualHome scene
        # Note: VirtualHome API may vary - this is a basic implementation
        # In practice, you'd use: self._scene = self._evolvinggraph.EvolvingGraph(...)
        # For now, we'll create a minimal stub that can be extended
        
        # Initialize default agent
        default_agent_id = "human"
        self._agents[default_agent_id] = {
            "room_id": "kitchen",  # Default starting room
            "position": (0.0, 0.0, 0.0),
            "held_objects": [],
        }
        
        # Initialize object locations (stub - would come from VirtualHome scene)
        self._object_locations_cache = {
            "knife": ObjectLocation("knife", None, "kitchen", (1.0, 0.0, 1.0)),
            "plate": ObjectLocation("plate", None, "kitchen", (2.0, 0.0, 1.0)),
            "fork": ObjectLocation("fork", None, "dining_room", (1.0, 0.0, 1.0)),
            "keys": ObjectLocation("keys", None, "bedroom", (1.0, 0.0, 1.0)),
            "book": ObjectLocation("book", None, "living_room", (1.0, 0.0, 1.0)),
        }
        
        # Return initial observation
        return self.get_visible_state(default_agent_id)

    def step(
        self, action: Action, agent_id: str
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute action and advance environment.

        Args:
            action: Action to execute
            agent_id: ID of agent taking action

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.timestep += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        
        agent = self._agents[agent_id]
        
        # Execute action based on type
        if action == Action.MOVE:
            # Move agent to adjacent room (simplified)
            current_room = agent["room_id"]
            # Simple movement logic - in practice would use VirtualHome navigation
            # For now, just update position slightly
            pos = agent["position"]
            agent["position"] = (pos[0] + 0.1, pos[1], pos[2])
            info["action"] = "moved"
        
        elif action == Action.PICKUP:
            # Pick up object if visible and nearby
            visible_objects = self._get_visible_objects(agent_id)
            if visible_objects:
                obj_id = visible_objects[0]  # Pick first visible object
                if obj_id in self._object_locations_cache:
                    agent["held_objects"].append(obj_id)
                    # Remove from scene (simplified)
                    info["object_picked_up"] = obj_id
        
        elif action == Action.PLACE:
            # Place held object at current location
            if agent["held_objects"]:
                obj_id = agent["held_objects"].pop(0)
                room_id = agent["room_id"]
                pos = agent["position"]
                self._object_locations_cache[obj_id] = ObjectLocation(
                    obj_id, None, room_id, pos
                )
                info["object_placed"] = obj_id
        
        elif action == Action.OPEN:
            # Open container (simplified - would interact with VirtualHome containers)
            info["action"] = "opened_container"
        
        elif action == Action.CLOSE:
            # Close container
            info["action"] = "closed_container"
        
        elif action == Action.WAIT:
            # Do nothing
            info["action"] = "waited"
        
        elif action == Action.SAY:
            # Communication action
            info["action"] = "communicated"
        
        # Get observation after action
        observation = self.get_visible_state(agent_id)
        
        # Check if episode should end (simplified - max steps)
        if self.timestep >= 100:
            done = True
        
        return observation, reward, done, info

    def get_true_state(self) -> Dict[str, Any]:
        """Get true state of the environment (privileged information).

        Returns:
            Dictionary containing true state information
        """
        return {
            "timestep": self.timestep,
            "agents": {
                agent_id: {
                    "room_id": agent["room_id"],
                    "position": agent["position"],
                    "held_objects": agent["held_objects"].copy(),
                }
                for agent_id, agent in self._agents.items()
            },
            "object_locations": {
                obj_id: {
                    "room_id": loc.room_id,
                    "container_id": loc.container_id,
                    "position": loc.position,
                }
                for obj_id, loc in self._object_locations_cache.items()
            },
        }

    def get_visible_state(self, agent_id: str) -> Observation:
        """Get visible state for an agent (partial observability).

        Args:
            agent_id: ID of agent

        Returns:
            Observation available to agent
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self._agents[agent_id]
        room_id = agent["room_id"]
        
        # Get visible objects (objects in same room)
        visible_objects = self._get_visible_objects(agent_id)
        
        # Get visible containers (containers in same room)
        visible_containers = self._get_visible_containers(agent_id)
        
        return Observation(
            agent_id=agent_id,
            visible_objects=visible_objects,
            visible_containers=visible_containers,
            current_room=room_id,
            position=agent["position"],
            timestamp=self.timestep,
        )

    def get_object_locations(self) -> Dict[str, ObjectLocation]:
        """Get current object locations.

        Returns:
            Dictionary mapping object_id to ObjectLocation
        """
        return self._object_locations_cache.copy()

    def _get_visible_objects(self, agent_id: str) -> List[str]:
        """Get list of visible object IDs for an agent.

        Args:
            agent_id: ID of agent

        Returns:
            List of visible object IDs
        """
        if agent_id not in self._agents:
            return []
        
        agent = self._agents[agent_id]
        room_id = agent["room_id"]
        
        # Objects are visible if in same room (simplified visibility model)
        visible = []
        for obj_id, loc in self._object_locations_cache.items():
            if loc.room_id == room_id:
                # Check if agent is holding it
                if obj_id not in agent["held_objects"]:
                    visible.append(obj_id)
        
        return visible

    def _get_visible_containers(self, agent_id: str) -> List[str]:
        """Get list of visible container IDs for an agent.

        Args:
            agent_id: ID of agent

        Returns:
            List of visible container IDs
        """
        if agent_id not in self._agents:
            return []
        
        agent = self._agents[agent_id]
        room_id = agent["room_id"]
        
        # Containers are visible if in same room
        # For now, return empty list (containers would come from VirtualHome scene)
        # In practice, would query VirtualHome scene for containers in room
        return []
