"""GridHouse: Symbolic grid-based household simulator."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from ..base import Environment
from ...common.types import Action, Observation, ObjectLocation
from ...common.seeding import get_rng


@dataclass
class Room:
    """Room definition."""

    room_id: str
    name: str
    bounds: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    containers: List[str] = field(default_factory=list)


@dataclass
class Container:
    """Container definition."""

    container_id: str
    name: str
    room_id: str
    position: Tuple[int, int]
    is_open: bool = False
    contents: List[str] = field(default_factory=list)  # Object IDs


class GridHouseEnvironment(Environment):
    """Symbolic grid-based household simulator.

    Provides a self-contained fallback when VirtualHome is not available.
    Implements partial observability via visibility radius and line-of-sight.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        visibility_radius: int = 5,
        rooms: Optional[List[Room]] = None,
        containers: Optional[List[Container]] = None,
        objects: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize GridHouse environment.

        Args:
            grid_size: Size of the grid (width, height)
            visibility_radius: Maximum visibility radius for agents
            rooms: List of rooms (if None, creates default rooms)
            containers: List of containers (if None, creates default containers)
            objects: List of object IDs (if None, creates default objects)
            seed: Random seed
        """
        self.grid_size = grid_size
        self.visibility_radius = visibility_radius
        self.rng = get_rng(seed)

        # Initialize rooms
        if rooms is None:
            self.rooms = self._create_default_rooms()
        else:
            self.rooms = {room.room_id: room for room in rooms}

        # Initialize containers
        if containers is None:
            self.containers = self._create_default_containers()
        else:
            self.containers = {c.container_id: c for c in containers}

        # Initialize objects
        if objects is None:
            self.objects = ["knife", "plate", "fork", "keys", "book"]
        else:
            self.objects = objects

        # Object locations: object_id -> ObjectLocation
        self.object_locations: Dict[str, ObjectLocation] = {}

        # Agent positions: agent_id -> (x, y)
        self.agent_positions: Dict[str, Tuple[int, int]] = {}

        # Agent rooms: agent_id -> room_id
        self.agent_rooms: Dict[str, str] = {}

        # Timestep
        self.timestep = 0

        # Initialize object locations randomly
        self._initialize_object_locations()

    def _create_default_rooms(self) -> Dict[str, Room]:
        """Create default room layout."""
        rooms = [
            Room("kitchen", "Kitchen", (0, 0, 10, 10)),
            Room("living_room", "Living Room", (10, 0, 20, 10)),
            Room("bedroom", "Bedroom", (0, 10, 10, 20)),
            Room("bathroom", "Bathroom", (10, 10, 20, 20)),
        ]
        return {room.room_id: room for room in rooms}

    def _create_default_containers(self) -> Dict[str, Container]:
        """Create default containers."""
        containers = [
            Container("cabinet_kitchen", "Kitchen Cabinet", "kitchen", (5, 5)),
            Container("drawer_kitchen", "Kitchen Drawer", "kitchen", (7, 5)),
            Container("table_living", "Living Room Table", "living_room", (15, 5)),
            Container("desk_bedroom", "Bedroom Desk", "bedroom", (5, 15)),
        ]
        return {c.container_id: c for c in containers}

    def _initialize_object_locations(self) -> None:
        """Initialize object locations randomly."""
        for obj_id in self.objects:
            # Randomly place in a container or room
            if self.rng.random() < 0.7:  # 70% in containers
                container_id = self.rng.choice(list(self.containers.keys()))
                container = self.containers[container_id]
                room_id = container.room_id
                position = (container.position[0] + self.rng.integers(-1, 2), container.position[1] + self.rng.integers(-1, 2))
                self.object_locations[obj_id] = ObjectLocation(
                    object_id=obj_id,
                    container_id=container_id,
                    room_id=room_id,
                    position=(float(position[0]), float(position[1]), 0.0),
                )
                container.contents.append(obj_id)
            else:  # 30% in rooms
                room_id = self.rng.choice(list(self.rooms.keys()))
                room = self.rooms[room_id]
                x = self.rng.integers(room.bounds[0] + 1, room.bounds[2] - 1)
                y = self.rng.integers(room.bounds[1] + 1, room.bounds[3] - 1)
                self.object_locations[obj_id] = ObjectLocation(
                    object_id=obj_id,
                    container_id=None,
                    room_id=room_id,
                    position=(float(x), float(y), 0.0),
                )

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = get_rng(seed)

        self.timestep = 0
        self.object_locations.clear()
        self.agent_positions.clear()
        self.agent_rooms.clear()

        # Reset containers
        for container in self.containers.values():
            container.is_open = False
            container.contents.clear()

        # Reinitialize object locations
        self._initialize_object_locations()

        # Place agents in starting positions (kitchen)
        self.agent_positions["human"] = (5, 5)
        self.agent_rooms["human"] = "kitchen"
        self.agent_positions["helper"] = (15, 5)
        self.agent_rooms["helper"] = "living_room"

        return self.get_visible_state("human")

    def step(self, action: Action, agent_id: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute action and advance environment."""
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if agent_id not in self.agent_positions:
            raise ValueError(f"Agent {agent_id} not found")

        pos = self.agent_positions[agent_id]
        room_id = self.agent_rooms[agent_id]

        # Execute action
        if action == Action.MOVE:
            # Simple movement (can be extended with direction)
            dx, dy = self.rng.integers(-1, 2, size=2)
            new_pos = (pos[0] + dx, pos[1] + dy)
            # Check bounds and room boundaries
            if self._is_valid_position(new_pos):
                self.agent_positions[agent_id] = new_pos
                # Update room if crossed boundary
                new_room = self._get_room_at_position(new_pos)
                if new_room:
                    self.agent_rooms[agent_id] = new_room

        elif action == Action.OPEN:
            # Open nearby container
            container = self._get_nearby_container(pos, room_id)
            if container and not container.is_open:
                container.is_open = True
                info["container_opened"] = container.container_id

        elif action == Action.CLOSE:
            # Close nearby container
            container = self._get_nearby_container(pos, room_id)
            if container and container.is_open:
                container.is_open = False
                info["container_closed"] = container.container_id

        elif action == Action.PICKUP:
            # Pickup nearby object
            obj = self._get_nearby_object(pos, room_id)
            if obj:
                # Remove from location
                if obj in self.object_locations:
                    loc = self.object_locations[obj]
                    if loc.container_id:
                        self.containers[loc.container_id].contents.remove(obj)
                    del self.object_locations[obj]
                info["object_picked_up"] = obj

        elif action == Action.PLACE:
            # Place object (simplified - requires tracking held objects)
            pass  # TODO: Implement object holding

        elif action == Action.WAIT:
            # Do nothing
            pass

        elif action == Action.SAY:
            # Communication action
            info["communication"] = True

        self.timestep += 1

        obs = self.get_visible_state(agent_id)
        return obs, reward, done, info

    def get_true_state(self) -> Dict[str, Any]:
        """Get true state of the environment."""
        return {
            "object_locations": self.object_locations.copy(),
            "agent_positions": self.agent_positions.copy(),
            "agent_rooms": self.agent_rooms.copy(),
            "containers": {
                cid: {
                    "is_open": c.is_open,
                    "contents": c.contents.copy(),
                }
                for cid, c in self.containers.items()
            },
            "timestep": self.timestep,
        }

    def get_visible_state(self, agent_id: str) -> Observation:
        """Get visible state for an agent (partial observability)."""
        if agent_id not in self.agent_positions:
            raise ValueError(f"Agent {agent_id} not found")

        pos = self.agent_positions[agent_id]
        room_id = self.agent_rooms[agent_id]

        # Find visible objects (within radius and line-of-sight)
        visible_objects = []
        visible_containers = []

        # Objects in same room within visibility radius
        for obj_id, obj_loc in self.object_locations.items():
            if obj_loc.room_id == room_id:
                distance = np.sqrt(
                    (obj_loc.position[0] - pos[0]) ** 2 + (obj_loc.position[1] - pos[1]) ** 2
                )
                if distance <= self.visibility_radius:
                    # Check if container is open (if object is in container)
                    if obj_loc.container_id:
                        container = self.containers[obj_loc.container_id]
                        if container.is_open:
                            visible_objects.append(obj_id)
                    else:
                        visible_objects.append(obj_id)

        # Containers in same room within visibility radius
        for container_id, container in self.containers.items():
            if container.room_id == room_id:
                distance = np.sqrt(
                    (container.position[0] - pos[0]) ** 2 + (container.position[1] - pos[1]) ** 2
                )
                if distance <= self.visibility_radius:
                    visible_containers.append(container_id)

        return Observation(
            agent_id=agent_id,
            visible_objects=visible_objects,
            visible_containers=visible_containers,
            current_room=room_id,
            position=(float(pos[0]), float(pos[1]), 0.0),
            timestamp=self.timestep,
        )

    def get_object_locations(self) -> Dict[str, ObjectLocation]:
        """Get current object locations."""
        return self.object_locations.copy()

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid."""
        x, y = pos
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def _get_room_at_position(self, pos: Tuple[int, int]) -> Optional[str]:
        """Get room ID at position."""
        x, y = pos
        for room_id, room in self.rooms.items():
            if room.bounds[0] <= x < room.bounds[2] and room.bounds[1] <= y < room.bounds[3]:
                return room_id
        return None

    def _get_nearby_container(self, pos: Tuple[int, int], room_id: str) -> Optional[Container]:
        """Get nearby container in room."""
        for container in self.containers.values():
            if container.room_id == room_id:
                distance = np.sqrt(
                    (container.position[0] - pos[0]) ** 2 + (container.position[1] - pos[1]) ** 2
                )
                if distance <= 2:  # Adjacent
                    return container
        return None

    def _get_nearby_object(self, pos: Tuple[int, int], room_id: str) -> Optional[str]:
        """Get nearby object in room."""
        for obj_id, obj_loc in self.object_locations.items():
            if obj_loc.room_id == room_id:
                distance = np.sqrt(
                    (obj_loc.position[0] - pos[0]) ** 2 + (obj_loc.position[1] - pos[1]) ** 2
                )
                if distance <= 1:  # Adjacent
                    return obj_id
        return None
