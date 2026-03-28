"""Symbolic VirtualHome Environment."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..base import Environment
from ...common.types import Action, Observation, ObjectLocation
from ...common.seeding import get_rng
from .graph import VHGraph, VHNode, VHRoom
from .layouts import create_layout

class VirtualHomeEnvironment(Environment):
    """High-fidelity symbolic VirtualHome environment.
    
    Simulates graph state dynamics without Unity overhead.
    Enforces partial observability via room-level occlusion.
    """
    
    def __init__(self, 
                 layout_id: int = 0, 
                 seed: Optional[int] = None,
                 objects: Optional[List[str]] = None):
        self.rng = get_rng(seed)
        self.graph = create_layout(layout_id, self.rng)
        
        # Agents: id -> room_id
        self.agent_locations: Dict[str, int] = {}
        
        # Populate objects if provided
        if objects:
            self._place_objects(objects)
            
        self.timestep = 0
        
        # Cache room mapping for fast lookup
        self.room_ids = {n.name: n.id for n in self.graph.nodes.values() if isinstance(n, VHRoom)}
        self.room_names = {v: k for k, v in self.room_ids.items()}
        
    def _place_objects(self, object_names: List[str]):
        """Randomly distribute objects in containers."""
        # Find all valid containers
        containers = [nid for nid, node in self.graph.nodes.items() 
                     if hasattr(node, 'can_open')] # Simple check for containers
        
        # ID generator start from max existing
        next_id = max(self.graph.nodes.keys()) + 1
        
        self.object_id_map = {} # name -> id
        
        for name in object_names:
            # Create object node
            from .graph import VHObject
            obj = VHObject(next_id, name, name)
            self.graph.add_node(obj)
            self.object_id_map[name] = next_id
            
            # Place in random container
            cid = self.rng.choice(containers)
            self.graph.add_edge(cid, obj.id)
            
            next_id += 1

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.rng = get_rng(seed)
            # Recreate graph? For now assume layout static, just shuffle objects/agents
        
        # Agents start in random rooms or fixed? Let's say Kitchen/Living
        self.agent_locations['human'] = self.room_ids['kitchen']
        self.agent_locations['helper'] = self.room_ids['living_room']
        self.timestep = 0
        
        return self.get_visible_state('human')

    def step(self, action: Action, agent_id: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute symbolic action."""
        info = {}
        curr_room_id = self.agent_locations[agent_id]
        
        if action == Action.MOVE:
            # Move to adjacent room (simplified: random jump to any other room for now 
            # or strictly connected? Let's allow full graph traversal for simplicity/speed)
            all_rooms = list(self.room_ids.values())
            # For realism, should be connected. But symbolic VH often allows walk_to(room).
            # We pick a random OTHER room for "MOVE" action unless specified?
            # Standard interface doesn't specify target. So we just stay or move random?
            # Better: In GridHouse MOVE is random walk. Here, let's random walk to another room.
            possible_rooms = [r for r in all_rooms if r != curr_room_id]
            if possible_rooms:
                new_room = self.rng.choice(possible_rooms)
                self.agent_locations[agent_id] = new_room
                
        elif action == Action.OPEN:
            # Open a container in current room
            # Pick a closed container in current room
            pass # Implementation detail
            
        elif action == Action.PICKUP:
            # Pick up object in current room
            pass
            
        self.timestep += 1
        obs = self.get_visible_state(agent_id)
        return obs, 0.0, False, info

    def get_visible_state(self, agent_id: str) -> Observation:
        """Get observation for agent (Partial Observability)."""
        room_id = self.agent_locations[agent_id]
        
        visible_objects = []
        visible_containers = []
        
        # Traverse graph starting from room_id
        # items in room directly, or in OPEN containers in room
        
        # 1. Direct children of room
        room_children = self.graph.children_map.get(room_id, set())
        
        for child_id in room_children:
            node = self.graph.nodes[child_id]
            # If container
            if hasattr(node, 'is_open'):
                visible_containers.append(node.name) # Use name as ID for agent consistency
                # If open, see contents
                if getattr(node, 'is_open', True) or getattr(node, 'can_open', False) == False: # Surface vs Container
                    # Surfaces (tables) always show contents. Closed drawers don't.
                    # Simplified logic: can_open=False means surface -> visible.
                    # is_open=True -> visible.
                    if (not node.can_open) or node.is_open:
                         contents = self.graph.children_map.get(child_id, set())
                         for obj_id in contents:
                             visible_objects.append(self.graph.nodes[obj_id].name)
            
            # If object directly in room (floor)
            if hasattr(node, 'properties'): # Is object
                visible_objects.append(node.name)

        return Observation(
            agent_id=agent_id,
            visible_objects=visible_objects,
            visible_containers=visible_containers,
            current_room=self.room_names[room_id],
            position=(0.0, 0.0, 0.0), # Topological, no coordinates
            timestamp=self.timestep
        )

    def get_true_state(self) -> Dict[str, Any]:
        """Return the true state of the environment."""
        return {
            "object_locations": self.get_object_locations(),
            "agent_locations": self.agent_locations.copy(),
            "timestep": self.timestep
        }

    def get_object_locations(self) -> Dict[str, ObjectLocation]:
        """Return true state for evaluation."""
        locs = {}
        for name, oid in getattr(self, 'object_id_map', {}).items():
            # Find parent room
            room_id = self.graph.get_room(oid)
            room_name = self.room_names.get(room_id, "unknown")
            parent_id = self.graph.parent_map.get(oid)
            parent_name = self.graph.nodes[parent_id].name if parent_id else None
            
            locs[name] = ObjectLocation(
                object_id=name,
                room_id=room_name,
                container_id=parent_name if parent_name != room_name else None,
                position=(0.0, 0.0, 0.0)
            )
        return locs
